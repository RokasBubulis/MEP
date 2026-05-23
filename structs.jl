include("generators.jl")
include("lie_algebra.jl")

using ForwardDiff

struct Algebra{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    structure_tensor::Array{Float64, 3}
    im_control_lie::Vector{T}
    neg_im_drift_lie::Vector{T}
end

function Algebra(im_control::SparseMatrixCSC{T, Int}, im_drift::SparseMatrixCSC{T, Int})
    lie_basis = construct_lie_basis_general([copy(im_control), copy(im_drift)])
    p_basis = lie_basis[2:end]
    structure_tensor = build_structure_tensor(lie_basis)
    im_control_lie = zeros(T, length(lie_basis))
    neg_im_drift_lie = zeros(T, length(lie_basis))
    project_to_algebra!(im_control_lie, im_control, algebra, stor)
    project_to_algebra!(neg_im_drift_lie, -im_drift, algebra, stor)
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
    period_im_control::Real

    function System{T}(im_control, im_drift, target, target_logic) where T
        eig = abs(eigvals(Matrix(im_control))[1])
        @assert eig != 0.0 "Control period eigenvalue assumption failed"
        new{T}(im_control, im_drift, target, target_logic, sparse(adjoint(target)), sparse(adjoint(target_logic)), 
        diag(im_control), diag(im_control)[[1,2,4,5]], 2*pi/eig)
    end 
end 

struct SolverParams
    tmax::Float64
    dt::Float64
    tol::Float64
    Newton_steps::Int64
    Newton_tol::Float64
    Newton_damping::Float64
end 

mutable struct Storage{T, R}

    # state control
    alpha::R
    # state matrices
    U0::Matrix{ComplexF64}

    tmp_logic::Matrix{ComplexF64}
    U_logic::Matrix{ComplexF64}

    # never dual — Newton loop scratch (only used to find α)
    tmp_adjoint_drift::Matrix{ComplexF64}
    tmp_adjoint_drift_1st_der::Matrix{ComplexF64}
    tmp_adjoint_drift_2nd_der::Matrix{ComplexF64}
    tmp_adjoint_drift_obj::Matrix{ComplexF64}
    tmp_adjoint_drift_1st_der_obj::Matrix{ComplexF64}
    tmp_adjoint_drift_2nd_der_obj::Matrix{ComplexF64}
    tmp_primal_costate::Matrix{ComplexF64}

    # output/intermediate matrices
    tmp::Matrix{ComplexF64}
    tmp1::Matrix{ComplexF64}; tmp2::Matrix{ComplexF64}; tmp3::Matrix{ComplexF64}
    tmp1_adj::Matrix{ComplexF64}

    # dual
    M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}; M::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}
    tmp_adjoint_drift_1st_der_obj_dual::Matrix{T}

        # output/intermediate matrices
    tmp_dual::Matrix{T}
    tmp1_dual::Matrix{T}; tmp2_dual::Matrix{T}; tmp3_dual::Matrix{T}

    # project algebra tmp 
    proj_alg_tmp::Matrix{T}

    # scratch for adjoint_action_by_campbell dual non dual versions
    campbell_array1::Vector{ComplexF64}; campbell_array2::Vector{ComplexF64}
    campbell_array3::Vector{ComplexF64}; campbell_array4::Vector{ComplexF64}; campbell_array5::Vector{ComplexF64}

    # scratch for bracket_via_lie_coeffs (exclusive) non dual versions
    bracket_array1::Vector{ComplexF64}; bracket_array2::Vector{ComplexF64}; bracket_array3::Vector{ComplexF64}

    # RK4 temps
    adjoint_drift_arr::Vector{ComplexF64}
    M_arr::Vector{ComplexF64}
    k1_arr::Vector{ComplexF64}
    k2_arr::Vector{ComplexF64}
    k3_arr::Vector{ComplexF64}
    k4_arr::Vector{ComplexF64}
    tmp_adj_drift_arr::Vector{ComplexF64}
    tmp_adj_drift1_arr::Vector{ComplexF64}
    tmp_adj_drift2_arr::Vector{ComplexF64}
    tmp_adj_drift3_arr::Vector{ComplexF64}
    tmp_M1_arr::Vector{ComplexF64}
    tmp_M2_arr::Vector{ComplexF64}
    tmp_M3_arr::Vector{ComplexF64}

    # dual versions for tmp arrays
    bracket_array1_dual::Vector{T}; bracket_array2_dual::Vector{T}; bracket_array3_dual::Vector{T}
    campbell_array1_dual::Vector{T}; campbell_array2_dual::Vector{T}
    campbell_array3_dual::Vector{T}; campbell_array4_dual::Vector{T}; campbell_array5_dual::Vector{T}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)),
    Matrix{ComplexF64}(I, dim, dim), # U0
    (Matrix{Complex}(undef, 4, 4) for _ in 1:2)...,
    (Matrix{ComplexF64}(undef, dim, dim) for _ in 1:12)...,  # Newton loop tmps for alpha
    (Matrix{T}(undef, dim, dim) for _ in 1:14)...,
    (Vector{ComplexF64}(undef, n_basis) for _ in 1:8)..., # Campbell formula tmp arrays + non dual bracket tmps
    (Vector{ComplexF64}(undef, n_basis) for _ in 1:13)..., # pbasis coefficients for M, H_opt, use full lie
    (Vector{T}(undef, n_basis) for _ in 1:8)...
)

primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)
primal(x::Complex{<:ForwardDiff.Dual}) = Complex(ForwardDiff.value(real(x)), ForwardDiff.value(imag(x)))
primal(x::Complex) = x
primal(x::Real) = x
primal(A::AbstractArray) = primal.(A)