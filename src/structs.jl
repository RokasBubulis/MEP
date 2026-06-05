include("generators.jl")
include("lie_algebra.jl")

using ExponentialUtilities

function tensor_basis(n_particles::Int, n_levels::Int)
    e = [I(n_levels)[:, 1], I(n_levels)[:, 2]]
    return hcat([reduce(kron, [e[b] for b in idx])
                 for idx in Iterators.product(fill(1:2, n_particles)...)]...)
end

function project_from_3_to_2_levels_for_control(U, ryd_to_logic_conv_mat)
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

    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
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
        lie_basis, p_basis, structure_tensor, adj_repr_map, im_control_lie, neg_im_drift_lie,
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

mutable struct Storage{T, R}
    # state control
    alpha::R

    # initial states Hilbert, size 3^n x 3^n
    U0::Matrix{T}

    # temporary matrices Hilbert, size 3^n x 3^n
    U::Matrix{T}
    dU::Matrix{T}
    H_opt::Matrix{T}
    tmp::Matrix{T}
    H_opt_dt::Matrix{T}
    U_buffer::Matrix{T}
    U_unitary_buffer_check::Matrix{T}
    tmp_3levels::Matrix{T}
    U_adj_buffer::Matrix{T}
    
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
    exp_cache::Tuple{Vector{Matrix{ComplexF64}}, Vector{Float64}}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)), # alpha
    Matrix{T}(I, dim, dim), # U0
    (Matrix{T}(undef, dim, dim) for _ in 1:9)..., # temporary matrices Hilbert, size 3^n x 3^n
    (Matrix{T}(undef, 4, 4) for _ in 1:2)..., # temporary matrices Hilbert logical, size 2^n x 2^n
    Matrix{real(T)}(undef, n_basis, n_basis), # temporary matrices Lie, size dim(g) x dim(g)
    (Vector{real(T)}(undef, n_basis) for _ in 1:4)..., # arrays Lie, size dim(g) x 1
    Vector{T}(undef, n_basis), # tmp_expv
    (Vector{T}(undef, 4) for _ in 1:2)..., # temporary arrays for distance 2 levels, size 2^n x 1
    (Vector{T}(undef, dim) for _ in 1:2)..., # temporary arrays for distance 3 levels, size 3^n x 1
    Matrix{T}(undef, dim, 4), # temporary matrix for conversion from 3 to 2 levels, size  3^n x 2^n

    ExpvCache{Float64}(dim), # m
    ExpMethodHigham2005(), 
    ExponentialUtilities.alloc_mem(Matrix{T}(undef, dim, dim), ExpMethodHigham2005()),
)
