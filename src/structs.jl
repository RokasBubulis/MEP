include("generators.jl")
include("lie_algebra.jl")

struct Algebra{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    structure_tensor::Array{Float64, 3}
    im_control_lie::Vector{Float64}
    neg_im_drift_lie::Vector{Float64}
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    structure_tensor = build_structure_tensor(lie_basis)
    im_control_lie = project_algebra(im_control, lie_basis)
    neg_im_drift_lie = project_algebra(-im_drift, lie_basis)
    return Algebra{T}(lie_basis, p_basis, structure_tensor, im_control_lie, neg_im_drift_lie)
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

    function System{T}(im_control, im_drift, target, target_logic) where T
        eig = abs(eigvals(Matrix(im_control))[1])
        @assert eig != 0.0 "Control period eigenvalue assumption failed"
        new{T}(im_control, im_drift, target, target_logic, sparse(adjoint(target)), sparse(adjoint(target_logic)), 
        diag(im_control), diag(im_control)[[1,2,4,5]])
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
    H_opt_lie::Vector{R}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)), 
    Matrix{T}(I, dim, dim), # U0
    (Matrix{T}(undef, dim, dim) for _ in 1:2)...,
    (Matrix{T}(undef, 4, 4) for _ in 1:2)...,
    Matrix{real(T)}(undef, n_basis, n_basis), 
    (Vector{real(T)}(undef, n_basis) for _ in 1:2)...,
)
