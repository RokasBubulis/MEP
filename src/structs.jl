include("generators.jl")
include("lie_algebra.jl")

struct Algebra{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    structure_tensor::Array{Float64, 3}
    adj_repr_map::Array{Float64, 2}
    im_control_lie::Vector{Float64}
    neg_im_drift_lie::Vector{Float64}
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    structure_tensor = build_structure_tensor(lie_basis)
    im_control_lie = project_algebra(im_control, lie_basis)
    adj_repr_map = build_adjoint_representation_map(im_control_lie, structure_tensor)
    neg_im_drift_lie = project_algebra(-im_drift, lie_basis)
    return Algebra{T}(lie_basis, p_basis, structure_tensor, adj_repr_map, im_control_lie, neg_im_drift_lie)
end 

function tensor_basis(n_particles::Int, n_levels::Int)
    e = [I(n_levels)[:, 1], I(n_levels)[:, 2]]
    return hcat([reduce(kron, [e[b] for b in idx])
                 for idx in Iterators.product(fill(1:2, n_particles)...)]...)
end

struct System{T}
    im_control::SparseMatrixCSC{T, Int}
    im_drift::SparseMatrixCSC{T, Int}
    target::SparseMatrixCSC{T, Int}
    target_logic::SparseMatrixCSC{T, Int}
    adjoint_target::SparseMatrixCSC{T, Int}
    adjoint_target_logic::SparseMatrixCSC{T, Int}
    im_control_vec::Vector{T}
    im_control_vec_logic::Vector{T}
    ryd_to_logic_conv_mat::Matrix{T}
    ryd_to_logic_conv_mat_adj::Matrix{T}

    function System{T}(im_control, im_drift, target, target_logic) where T
        # assumes a control and drift in 3 level description
        n_particles = round(Int, (size(im_control,1)^(1/3)))
        Ψ = tensor_basis(n_particles, 3) 
        Φ = tensor_basis(n_particles, 2)
        ryd_to_logic_conv_mat = Ψ * Φ'

        new{T}(im_control, im_drift, target, target_logic, sparse(adjoint(target)), sparse(adjoint(target_logic)), 
        diag(im_control), diag(im_control)[[1,2,4,5]], ryd_to_logic_conv_mat, adjoint(ryd_to_logic_conv_mat))
    end 
end 

struct SolverParams
    tmax::Float64
    reltol::Float64
    abstol::Float64
    dist_tol::Float64
    opt_tol::Float64
end 

mutable struct MinDistanceTracker
    min_dist::Float64
end

mutable struct Storage{T, R}
    # state control
    alpha::R

    # initial states Hilbert
    U0::Matrix{T}

    # temporary matrices Hilbert
    H_opt::Matrix{T}
    tmp::Matrix{T}
    
    # temporary matrices Hilbert reduced logical
    tmp_logic::Matrix{T}
    U_logic::Matrix{T}

    # temporary matrices Lie 
    tmp_mat_lie::Matrix{R}

    # arrays Lie 
    tmp_adj_drift_arr::Vector{R}
    tmp_adj_drift_first_der_arr::Vector{R}
    tmp_adj_drift_second_der_arr::Vector{R}
    H_opt_lie::Vector{R}

    # temporary matrices for projection to logical subspace 
    tmp_logical1::Matrix{T}

end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)), 
    Matrix{T}(I, dim, dim), # U0
    (Matrix{T}(undef, dim, dim) for _ in 1:2)...,
    (Matrix{T}(undef, 4, 4) for _ in 1:2)...,
    Matrix{real(T)}(undef, n_basis, n_basis), 
    (Vector{real(T)}(undef, n_basis) for _ in 1:4)...,
    Matrix{T}(undef, dim, 4), 
)
