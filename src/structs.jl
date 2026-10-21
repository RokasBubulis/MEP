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
    KrylovSubspace::KrylovSubspace{Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}}
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

    KrylovSubspace = arnoldi(adj_repr_map, neg_im_drift_lie; m=length(lie_basis))
    
    return Algebra{T}(
        n_particles, n_levels, im_control, im_drift,
        lie_basis, p_basis, depths, structure_tensor, adj_repr_map, im_control_lie, neg_im_drift_lie,
        ryd_to_logic_conv_mat, adj_ryd_to_logic_conv_mat, 
        im_control_vec_2levels, im_control_vec_3levels, KrylovSubspace
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

mutable struct OutputTracker
    min_dist::Float64
    tstar::Float64
end

struct TargetContainer
    target::AbstractMatrix
    adjoint_target::AbstractMatrix
    tmp_for_target::Matrix{ComplexF64}
end 
function TargetContainer(target::AbstractMatrix)
    return TargetContainer(target, adjoint(target), Matrix{ComplexF64}(undef, size(target, 1), size(target, 1)))
end 

Base.@kwdef mutable struct Storage{T, R, C}
    # state control
    alpha::R
    a::R                        # -d²H_opt/dα² contracted with costate M at the optimal α,
                                # i.e. H_opt''^{α M_α} = ⟨∂²H_opt/∂α², M⟩

    # initial states Hilbert, size 3^n x 3^n
    U0::Matrix{T}

    # temporary matrices Hilbert, size 3^n x 3^n
    U::Matrix{T}
    dU::Matrix{T}
    H_opt::Matrix{T}
    tmp::Matrix{T}
    H_opt_prime::Matrix{T}      # ∂H_opt/∂α  (renamed from unused H_opt_dt)
    U_buffer::Matrix{T}
    U_unitary_buffer_check::Matrix{T}
    tmp_3levels::Matrix{T}
    U_adj_buffer::Matrix{T}

    # sensitivities
    χ::Matrix{T}
    B::Matrix{T}
    dχ::Matrix{T}
    dB::Matrix{T}
    # temporary matrices Hilbert logical, size 2^n x 2^n
    tmp_2levels::Matrix{T}
    U_2levels::Matrix{T}

    # temporary matrices Lie, size dim(g) x dim(g)
    tmp_mat_lie::Matrix{R}

    # arrays Lie, size dim(g) x 1
    tmp_adj_drift_arr::Vector{R}
    tmp_adj_drift_first_der_arr::Vector{R}
    tmp_adj_drift_second_der_arr::Vector{R}
    H_opt_lie::Vector{R}
    H_opt_prime_lie::Vector{R}    # ∂H_opt/∂α  in Lie representation
    H_opt_dprime_lie::Vector{R}   # ∂²H_opt/∂α² in Lie representation
    overlap_Hdprime_M::R
    tmp_expv::Vector{T}

    # temporary arrays for distance 2 levels, size 2^n x 1
    tmp_diag_2levels::Vector{T}
    exp_buffer_dist_2levels::Vector{T}

    # temporary arrays for distance 3 levels, size 3^n x 1
    tmp_diag_3levels::Vector{T}
    exp_buffer_dist_3levels::Vector{T}

    # temporary matrix for conversion from 3 to 2 levels, size  3^n x 2^n
    tmp_logical1::Matrix{T}

    expv_cache::ExpvCache{Float64}

    exp_method::ExpMethodHigham2005
    exp_cache::C
end

function Storage(n_particles::Int, n_basis::Int)
    T = ComplexF64
    R = real(T)
    ryd_dim = 3^n_particles
    log_dim = 2^n_particles

    exp_cache = ExponentialUtilities.alloc_mem(
        Matrix{T}(undef, ryd_dim, ryd_dim), ExpMethodHigham2005())

    return Storage(
        alpha         = zero(R),
        a             = one(R),
        U0            = Matrix{T}(I, ryd_dim, ryd_dim),
        U             = Matrix{T}(undef, ryd_dim, ryd_dim),
        dU            = Matrix{T}(undef, ryd_dim, ryd_dim),
        H_opt         = Matrix{T}(undef, ryd_dim, ryd_dim),
        tmp           = Matrix{T}(undef, ryd_dim, ryd_dim),
        H_opt_prime   = Matrix{T}(undef, ryd_dim, ryd_dim),
        U_buffer      = Matrix{T}(undef, ryd_dim, ryd_dim),
        U_unitary_buffer_check = Matrix{T}(undef, ryd_dim, ryd_dim),
        tmp_3levels   = Matrix{T}(undef, ryd_dim, ryd_dim),
        U_adj_buffer  = Matrix{T}(undef, ryd_dim, ryd_dim),
        χ = Matrix{T}(I, n_basis, n_basis),
        B = Matrix{T}(undef, n_basis, n_basis),
        dχ = Matrix{T}(undef, n_basis, n_basis),
        dB = Matrix{T}(undef, n_basis, n_basis),

        tmp_2levels   = Matrix{T}(undef, log_dim, log_dim),
        U_2levels     = Matrix{T}(undef, log_dim, log_dim),

        tmp_mat_lie   = Matrix{R}(undef, n_basis, n_basis),

        tmp_adj_drift_arr            = Vector{R}(undef, n_basis),
        tmp_adj_drift_first_der_arr  = Vector{R}(undef, n_basis),
        tmp_adj_drift_second_der_arr = Vector{R}(undef, n_basis),
        H_opt_lie                    = Vector{R}(undef, n_basis),
        H_opt_prime_lie              = Vector{R}(undef, n_basis),
        H_opt_dprime_lie             = Vector{R}(undef, n_basis),
        overlap_Hdprime_M = one(R), 
        tmp_expv                     = Vector{T}(undef, n_basis),

        tmp_diag_2levels        = Vector{T}(undef, log_dim),
        exp_buffer_dist_2levels = Vector{T}(undef, log_dim),

        tmp_diag_3levels        = Vector{T}(undef, ryd_dim),
        exp_buffer_dist_3levels = Vector{T}(undef, ryd_dim),

        tmp_logical1  = Matrix{T}(undef, ryd_dim, log_dim),

        expv_cache    = ExpvCache{Float64}(ryd_dim),
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
    return TargetContainer(target)
end 

function build_tar_ryd(type, phi, n)
    if type == "k0"
        ryd_target = gate_k0(n, phi)
    elseif type == "k1"
        ryd_target = gate_k1(n, phi)
    else
        throw("unknown gate type")
    end 
    return TargetContainer(ryd_target)
end 