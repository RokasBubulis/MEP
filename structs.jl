include("generators.jl")
include("lie_algebra.jl")

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

mutable struct Storage{T, R<:Real}
    # state control
    alpha::R
    # state matrices
    U0::Matrix{T}
    M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}

    # output/intermediate matrices
    tmp::Matrix{T}
    tmp1::Matrix{T}; tmp2::Matrix{T}; tmp3::Matrix{T}

    # project algebra tmp 
    proj_alg_tmp::Matrix{T}
    # adjoint drift tmp 
    tmp_adjoint_drift::Matrix{T}
    tmp_adjoint_drift_1st_der::Matrix{T}
    tmp_adjoint_drift_2nd_der::Matrix{T}
    # adjoint drift obj tmp 
    tmp_adjoint_drift_obj::Matrix{T}
    tmp_adjoint_drift_1st_der_obj::Matrix{T}
    tmp_adjoint_drift_2nd_der_obj::Matrix{T}

    # scratch for adjoint_action_by_campbell (exclusive)
    campbell_array1::Vector{T}; campbell_array2::Vector{T}
    campbell_array3::Vector{T}; campbell_array4::Vector{T}; campbell_array5::Vector{T}

    # scratch for bracket_via_lie_coeffs (exclusive)
    bracket_array1::Vector{T}; bracket_array2::Vector{T}; bracket_array3::Vector{T}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)),
    Matrix{T}(I, dim, dim),  # U0
    (Matrix{T}(undef, dim, dim) for _ in 1:18)...,
    (Vector{T}(undef, n_basis) for _ in 1:8)...
)