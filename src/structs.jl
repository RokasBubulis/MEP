include("generators.jl")
include("lie_algebra.jl")

using ExponentialUtilities

function tensor_basis(n_particles::Int, n_levels::Int)
    e = [I(n_levels)[:, 1], I(n_levels)[:, 2]]
    return hcat([reduce(kron, [e[b] for b in idx])
                 for idx in Iterators.product(fill(1:2, n_particles)...)]...)
end

function project_from_3_to_2_levels(U, ryd_to_logic_conv_mat)
    # U^(2) = Φ * Ψ' * U^(3) * Ψ * Φ' 
    # where Ψ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0,0), (0,1,0)}
    # Φ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0), (0,1)}
    # where n is the number of particles 
    # this allows projection from 3 level operator to 2 where |r> states are excluded

    return adjoint(ryd_to_logic_conv_mat) * U * ryd_to_logic_conv_mat
end

struct Algebra{T}
    n_particles::Int
    n_levels::Int
    im_control::SparseMatrixCSC{T, Int}
    im_drift::SparseMatrixCSC{T, Int}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    depths::Vector{Int}
    structure_tensor::Array{Float64, 3}
    adj_repr_map::Array{Float64, 2}
    im_control_lie::Vector{Float64}
    neg_im_drift_lie::Vector{Float64}
    ryd_to_logic_conv_mat::Matrix{T}
    adj_ryd_to_logic_conv_mat::Matrix{T}
    im_control_vec_2levels::Vector{T}
    im_control_vec_3levels::Vector{T}
    im_single_qubit_phase_vec::Vector{T}   # i·(number of qubits in |1⟩) per logical basis state: Z(θ)⊗…⊗Z(θ) = Diagonal(exp.(θ .* this))
    KrylovSubspace::KrylovSubspace{Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}}
    isometries::Vector{Matrix{T}}                        # V_λ: 3^N x 2, spans the λ-th invariant subspace (λ = 1 … N)
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    s = size(im_control, 1)
    n_levels = isinteger(log2(s)) ? 2 : 3
    @assert n_levels == 3
    n_particles = round(Int, log(n_levels, s))

    lie_basis, depths = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    structure_tensor = build_structure_tensor(lie_basis)
    im_control_lie = project_algebra(im_control, lie_basis)
    adj_repr_map = build_adjoint_representation_map(im_control_lie, structure_tensor)
    neg_im_drift_lie = project_algebra(-im_drift, lie_basis)

    Ψ = tensor_basis(n_particles, 3) 
    Φ = tensor_basis(n_particles, 2)
    ryd_to_logic_conv_mat = Ψ * Φ'
    adj_ryd_to_logic_conv_mat = adjoint(ryd_to_logic_conv_mat)

    im_control_vec_3levels = diag(im_control)
    im_control_2levels = adj_ryd_to_logic_conv_mat * im_control * ryd_to_logic_conv_mat
    im_control_vec_2levels = diag(im_control_2levels)

    # single-qubit phase rotations on the logical subspace: Z(θ) = diag(1, e^{iθ}) on every qubit,
    # Z(θ)⊗…⊗Z(θ) = exp(iθ Σ_j |1⟩⟨1|_j), so only the number of qubits in |1⟩ per basis state is needed.
    n1  = spdiagm(0 => T[0, 1, 0])
    id_ = spdiagm(0 => ones(T, 3))
    n_ones_ryd = sum(reduce(kron, [j == i ? n1 : id_ for j in 1:n_particles]) for i in 1:n_particles)
    im_single_qubit_phase_vec = im .* diag(adj_ryd_to_logic_conv_mat * n_ones_ryd * ryd_to_logic_conv_mat)

    KrylovSubspace = arnoldi(adj_repr_map, neg_im_drift_lie; m=length(lie_basis))

    isometries = build_Rydberg_isometries(n_particles)

    return Algebra{T}(
        n_particles, n_levels, im_control, im_drift,
        lie_basis, p_basis, depths, structure_tensor, adj_repr_map, im_control_lie, neg_im_drift_lie,
        ryd_to_logic_conv_mat, adj_ryd_to_logic_conv_mat, 
        im_control_vec_2levels, im_control_vec_3levels, im_single_qubit_phase_vec, KrylovSubspace,
        isometries
    )
end

struct SolverParams
    tmax::Float64
    reltol::Float64
    abstol::Float64
    dist_tol::Float64
    grad_tol::Float64
    unitary_tol::Float64
    tstar_min::Float64
    tstar_max::Float64
end 

mutable struct PropagationOutput
    # value and state of the propagation at t_star (t_star = T unless track_min)
    min_cost::Float64
    t_star::Float64
    U_t_star::Vector{Matrix{ComplexF64}}   # reduced propagators U^(λ) = V_λ' U V_λ
    B_t_star::Matrix{Float64}              # ∂U/∂m in the Lie basis
    M_t_star::Vector{Float64}
    χ_t_star::Matrix{Float64}
    H_t_star::Vector{Float64}              # H_opt(t_star) in the Lie basis
    a_t::Vector{ComplexF64}                # amplitudes a_r = (U^(r))[1,1] for r = 1 … N, preceded by a_0 = 1 (dark)
    τ::Vector{ComplexF64}                  # target amplitudes on the same subspaces
    phase_opt::Float64                     # θ_opt minimising φ(θ) at t_star
    target_k::Int                          # CkPhase order (1: all-ones gets the phase, 0: all-zeros)
    target_ϕ::Float64                      # controlled phase


    # scratch of cost_grad!: the full 3^N operator Σ_ν c_ν L_ν, its λ-th restriction and the
    # 2 x 3^N buffer of apply_λth_isometry_transformation!
    op_full::Matrix{ComplexF64}
    op_λ::Matrix{ComplexF64}
    iso_tmp::Matrix{ComplexF64}
end

Base.@kwdef mutable struct Storage{T, R, C}
    # state control
    alpha::R
    a::R
    phase_opt::R

    # N inequivalent two-level systems, Hilbert representation
    U::Vector{Matrix{T}}                       # reduced propagators U^(λ) on the invariant subspaces
    dU::Vector{Matrix{T}}                      # exp(dt H_mid^(λ)) for the λ-th subspace
    H_opt::Matrix{T}                           # H_opt(t) in the full 3^N x 3^N representation
    H_opt_t_minus_dt::Matrix{T}
    H_opt_invariant::Vector{Matrix{T}}         # H_opt(t) restricted to V_λ
    H_opt_t_minus_dt_invariant::Vector{Matrix{T}}
    H_opt_lie::Vector{R}                       # H_opt(t) in the Lie basis
    H_opt_t_minus_dt_lie::Vector{R}
    tmp_for_isometry::Matrix{T}                # scratch of apply_λth_isometry_transformation! (2 x 3^N)

    # target amplitudes on the invariant subspaces: index 1 is the dark |0…0⟩ state (r = 0 … N)
    target_amplitude::Vector{T}

    # sensitivities
    χ::Matrix{T}
    B::Matrix{T}
    dχ::Matrix{T}
    dB::Matrix{T}
    D::Matrix{R}
    E::Matrix{R}
    F::Matrix{R}

    # arrays Lie, size dim(g) x 1
    tmp_adj_drift_arr::Vector{R}
    tmp_adj_drift_first_der_arr::Vector{R}
    tmp_adj_drift_second_der_arr::Vector{R}
    H_opt_prime_lie::Vector{R}    # ∂H_opt/∂α  in Lie representation
    overlap_Hdprime_M::R
    bracket_lie::Vector{R}        # [H_opt', M] in Lie representation (used in costate_sensitivity_drift!)

    U_buffer::Vector{Matrix{T}}

    expv_cache::ExpvCache{Float64}

    exp_method::ExpMethodHigham2005
    exp_cache::C
end

function Storage(n_particles::Int, n_basis::Int)
    T = ComplexF64
    R = real(T)
    log_dim = 2^n_particles
    full_dim = 3^n_particles

    # the propagated objects are 2 x 2 (one per invariant subspace)
    exp_cache = ExponentialUtilities.alloc_mem(
        Matrix{T}(undef, 2, 2), ExpMethodHigham2005())

    return Storage(
        alpha         = zero(R),
        a             = one(R),
        phase_opt     = zero(R),
        U = [Matrix{T}(undef, 2, 2) for _ in 1:n_particles],
        dU = [Matrix{T}(undef, 2, 2) for _ in 1:n_particles],
        H_opt = Matrix{T}(undef, full_dim, full_dim),
        H_opt_t_minus_dt = Matrix{T}(undef, full_dim, full_dim),
        H_opt_invariant = [Matrix{T}(undef, 2, 2) for _ in 1:n_particles],
        H_opt_t_minus_dt_invariant = [Matrix{T}(undef, 2, 2) for _ in 1:n_particles],
        H_opt_lie = Vector{R}(undef, n_basis),
        H_opt_t_minus_dt_lie = Vector{R}(undef, n_basis),
        tmp_for_isometry = Matrix{T}(undef, 2, full_dim),

        target_amplitude = Vector{T}(undef, n_particles + 1),

        χ = Matrix{T}(I, n_basis, n_basis),
        B = Matrix{T}(undef, n_basis, n_basis),
        dχ = Matrix{T}(undef, n_basis, n_basis),
        dB = Matrix{T}(undef, n_basis, n_basis),
        D = Matrix{R}(undef, n_basis, n_basis),
        E = Matrix{R}(undef, n_basis, n_basis),
        F = Matrix{R}(undef, n_basis, n_basis),

        tmp_adj_drift_arr            = Vector{R}(undef, n_basis),
        tmp_adj_drift_first_der_arr  = Vector{R}(undef, n_basis),
        tmp_adj_drift_second_der_arr = Vector{R}(undef, n_basis),
        H_opt_prime_lie              = Vector{R}(undef, n_basis),
        overlap_Hdprime_M = one(R),
        bracket_lie                  = Vector{R}(undef, n_basis),

        U_buffer = [Matrix{T}(undef, 2, 2) for _ in 1:n_particles],

        expv_cache    = ExpvCache{Float64}(log_dim),
        exp_method    = ExpMethodHigham2005(),
        exp_cache     = exp_cache
    )
end

function build_tar(type, phi, n, algebra)
    if type == "k0"
        ryd_target = gate_k0(n, phi)
    elseif type == "k1"
        ryd_target = gate_k1(n, phi)
    else
        throw("unknown gate type")
    end 
    target = project_from_3_to_2_levels(ryd_target, algebra.ryd_to_logic_conv_mat)
    return TargetContainer(target, type == "k1" ? 1 : 0, Float64(phi))
end 

function build_tar_ryd(type, phi, n)
    if type == "k0"
        ryd_target = gate_k0(n, phi)
    elseif type == "k1"
        ryd_target = gate_k1(n, phi)
    else
        throw("unknown gate type")
    end 
    return TargetContainer(ryd_target, type == "k1" ? 1 : 0, Float64(phi))
end 

struct TargetContainer
    target::AbstractMatrix
    k::Int        # CkPhase order: k = 1 (all-ones state gets the phase) or k = 0 (all-zeros)
    ϕ::Float64    # controlled phase
end 