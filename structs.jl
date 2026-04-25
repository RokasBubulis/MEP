include("generators.jl")
include("lie_algebra.jl")

struct Algebra{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    return Algebra{T}(lie_basis, p_basis)
end 

struct System{T}
    im_control::SparseMatrixCSC{T, Int}
    im_drift::SparseMatrixCSC{T, Int}
    target::SparseMatrixCSC{T, Int}
    adjoint_target::SparseMatrixCSC{T, Int}
    im_control_vec::Vector{T}

    function System{T}(im_control, im_drift, target) where T
        new{T}(im_control, im_drift, target, sparse(adjoint(target)), diag(im_control))
    end 
end 

struct SolverParams
    tmax::Float64
    dt::Float64
    tol::Float64
    lambda::Float64
    Newton_steps::Int64
    Newton_tol::Float64
end 

mutable struct Storage{T}
    U0::Matrix{T}; M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}; tmp::Matrix{T}
end 

Storage{T}(dim::Int) where T = Storage{T}(
    Matrix{T}(I, dim, dim),
    (Matrix{T}(undef, dim, dim) for _ in 1:8)...
)
