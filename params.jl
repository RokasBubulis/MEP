include("generators.jl")
include("lie_algebra.jl")
T = ComplexF64

struct PhysicsParams{T}
    im_drift::SparseMatrixCSC{T,Int}
    im_control::SparseMatrixCSC{T,Int}
    target::SparseMatrixCSC{T,Int}
    adjoint_target::SparseMatrixCSC{T,Int}
end

struct AlgebraParams{T}
    lie_basis::Vector{SparseMatrixCSC{T,Int}}
    p_basis::Vector{SparseMatrixCSC{T,Int}}
end

AlgebraParams(p::PhysicsParams) = begin
    lie_basis = construct_lie_basis_general([copy(p.im_control), copy(p.im_drift)])
    AlgebraParams(lie_basis, lie_basis[2:end])
end

struct SolverParams
    tmax::Float64
    dt::Float64
    tol::Float64
end

struct Params{T}
    physics::PhysicsParams{T}
    algebra::AlgebraParams{T}
    solver::SolverParams
    U0::Matrix{T}
end

mutable struct StorageParams{T}
    M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}; tmp::Matrix{T}
end

StorageParams{T}(dim::Int) where T = StorageParams{T}(
    (Matrix{T}(undef, dim, dim) for _ in 1:8)...
)

### Specific test cases ### 

function prepare_2q_setup_with_target_from_Lie_coeffs(lie_coeffs::Vector{Float64}, tmax::Float64, dt::Float64, tol::Float64, T=ComplexF64)
    n_qubits = 2
    im_control, im_drift = im .* construct_Ryd_generators(n_qubits)

    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]

    @assert length(lie_coeffs) == length(lie_basis) "Number of coeffs must match Lie algebra dimension"
    target = sparse(exp(-Matrix(sum(lie_coeffs[i] * lie_basis[i] for i in eachindex(lie_coeffs)))))

    physics = PhysicsParams{T}(im_drift, im_control, target, sparse(adjoint(target)))
    algebra = AlgebraParams{T}(lie_basis, p_basis)
    solver  = SolverParams(tmax, dt, tol)

    dim = size(im_control, 1)
    params = Params{T}(physics, algebra, solver, Matrix{T}(I, dim, dim))
    stor   = StorageParams{T}(dim)

    return params, stor
end
