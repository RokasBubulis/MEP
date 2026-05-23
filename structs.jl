include("generators.jl")
include("lie_algebra.jl")

using ForwardDiff

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
    im_control_lie = project_to_lie_basis(im_control, lie_basis)
    neg_im_drift_lie = project_to_lie_basis(-im_drift, lie_basis)
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
    U0::Matrix{T}

    tmp_logic::Matrix{T}
    U_logic::Matrix{T}

    # Newton loop scratch (only used to find α)
    tmp_adjoint_drift::Matrix{T}
    tmp_adjoint_drift_1st_der::Matrix{T}
    tmp_adjoint_drift_2nd_der::Matrix{T}
    tmp_adjoint_drift_obj::Matrix{T}
    tmp_adjoint_drift_1st_der_obj::Matrix{T}
    tmp_adjoint_drift_2nd_der_obj::Matrix{T}

    M0::Matrix{T}; M1::Matrix{T}; M2::Matrix{T}; M::Matrix{T}
    U::Matrix{T}; dU::Matrix{T}; dM::Matrix{T}
    adjoint_drift::Matrix{T}
    # output/intermediate matrices
    tmp::Matrix{T}
    tmp1::Matrix{T}; tmp2::Matrix{T}; tmp3::Matrix{T}
    tmp1_adj::Matrix{T}
    # project algebra tmp 
    proj_alg_tmp::Matrix{T}

    # scratch for adjoint_action_by_campbell dual non dual versions
    campbell_array1::Vector{R}; campbell_array2::Vector{R}
    campbell_array3::Vector{R}; campbell_array4::Vector{R}; campbell_array5::Vector{R}

    # scratch for bracket_via_lie_coeffs (exclusive) non dual versions
    bracket_array1::Vector{R}; bracket_array2::Vector{R}; bracket_array3::Vector{R}

    # RK4 temps
    adjoint_drift_arr::Vector{R}
    M_arr::Vector{R}
    k1_arr::Vector{R}
    k2_arr::Vector{R}
    k3_arr::Vector{R}
    k4_arr::Vector{R}
    tmp_adj_drift_arr::Vector{R}
    tmp_adj_drift1_arr::Vector{R}
    tmp_adj_drift2_arr::Vector{R}
    tmp_adj_drift3_arr::Vector{R}
    tmp_M1_arr::Vector{R}
    tmp_M2_arr::Vector{R}
    tmp_M3_arr::Vector{R}
end

Storage{T}(dim::Int, n_basis::Int) where T = Storage{T, real(T)}(
    zero(real(T)),
    Matrix{T}(I, dim, dim), # U0
    (Matrix{T}(undef, 4, 4) for _ in 1:2)..., # logical subspace, TODO generalise dimension
    (Matrix{T}(undef, dim, dim) for _ in 1:20)...,  # remaining matrices
    (Vector{real(T)}(undef, n_basis) for _ in 1:21)...
)