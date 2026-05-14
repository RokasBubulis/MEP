include("generators.jl")
include("lie_algebra.jl")

using ForwardDiff

struct Algebra{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    structure_tensor::Array{Float64, 3}
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    structure_tensor = build_structure_tensor(lie_basis)
    return Algebra{T}(lie_basis, p_basis, structure_tensor)
end 

struct System{T}
    im_control::SparseMatrixCSC{T, Int}
    im_drift::SparseMatrixCSC{T, Int}
    target::SparseMatrixCSC{T, Int}
    adjoint_target::SparseMatrixCSC{T, Int}
    im_control_vec::Vector{T}
    period_im_control::Real

    function System{T}(im_control, im_drift, target) where T
        eig = abs(eigvals(Matrix(im_control))[1])
        @assert eig != 0.0 "Control period eigenvalue assumption failed"
        new{T}(im_control, im_drift, target, sparse(adjoint(target)), diag(im_control),2*pi/eig)
    end 
end 

struct SolverParams
    tmax::Float64
    dt::Float64
    tol::Float64
    lambda::Float64
    Newton_steps::Int64
    Newton_tol::Float64
    Newton_damping::Float64
end 

# mutable struct Storage{T}
#     U0::Matrix{T}; tmp_array1::Vector{T}; 
#     tmp_array2::Vector{T}; tmp_array3::Vector{T}; tmp_array4::Vector{T}; tmp_array5::Vector{T}
#     M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}
#     U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
#     adjoint_drift::Matrix{T}; tmp::Matrix{T}; 
#     tmp1::Matrix{T}; tmp2::Matrix{T}; tmp3::Matrix{T}
# end 

# Storage{T}(dim::Int) where T = Storage{T}(
#     Matrix{T}(I, dim, dim), 
#     (Vector{T}(undef, 8) for _ in 1:5)... ,
#     (Matrix{T}(undef, dim, dim) for _ in 1:11)...
# )

mutable struct Storage{T, R}

    # state control
    alpha::R
    # state matrices
    U0::Matrix{ComplexF64}

    # never dual — Newton loop scratch (only used to find α)
    tmp_adjoint_drift::Matrix{ComplexF64}
    tmp_adjoint_drift_1st_der::Matrix{ComplexF64}
    tmp_adjoint_drift_2nd_der::Matrix{ComplexF64}
    tmp_adjoint_drift_obj::Matrix{ComplexF64}
    tmp_adjoint_drift_1st_der_obj::Matrix{ComplexF64}
    tmp_adjoint_drift_2nd_der_obj::Matrix{ComplexF64}
    tmp_primal_costate::Matrix{ComplexF64}

    # dual
    M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}
    tmp_adjoint_drift_1st_der_obj_dual::Matrix{T}

    # output/intermediate matrices
    tmp::Matrix{T}
    tmp1::Matrix{T}; tmp2::Matrix{T}; tmp3::Matrix{T}

    # project algebra tmp 
    proj_alg_tmp::Matrix{T}

    # scratch for adjoint_action_by_campbell dual non dual versions
    campbell_array1::Vector{ComplexF64}; campbell_array2::Vector{ComplexF64}
    campbell_array3::Vector{ComplexF64}; campbell_array4::Vector{ComplexF64}; campbell_array5::Vector{ComplexF64}

    # scratch for bracket_via_lie_coeffs (exclusive) non dual versions
    bracket_array1::Vector{ComplexF64}; bracket_array2::Vector{ComplexF64}; bracket_array3::Vector{ComplexF64}

    # dual versions for tmp arrays
    bracket_array1_dual::Vector{T}; bracket_array2_dual::Vector{T}; bracket_array3_dual::Vector{T}
    campbell_array1_dual::Vector{T}; campbell_array2_dual::Vector{T}
    campbell_array3_dual::Vector{T}; campbell_array4_dual::Vector{T}; campbell_array5_dual::Vector{T}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)),
    Matrix{ComplexF64}(I, dim, dim), # U0
    (Matrix{ComplexF64}(undef, dim, dim) for _ in 1:7)...,  # Newton loop tmps for alpha
    (Matrix{T}(undef, dim, dim) for _ in 1:13)...,
    (Vector{ComplexF64}(undef, n_basis) for _ in 1:8)..., # Campbell formula tmp arrays + non dual bracket tmps
    (Vector{T}(undef, n_basis) for _ in 1:8)...
)

primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)
primal(x::Complex{<:ForwardDiff.Dual}) = Complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
primal(x::Complex) = x
primal(x::Real) = x
primal(A::AbstractArray) = primal.(A)