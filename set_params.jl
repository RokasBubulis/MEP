include("generators.jl")
include("lie_algebra.jl")
T = ComplexF64

struct SystemParams{T}
    im_drift::SparseMatrixCSC{T, Int} # i H_0
    im_control::SparseMatrixCSC{T, Int}  # i sum_j Z_j
    target::SparseMatrixCSC{T, Int}  # U_T 
end

struct DerivedArgs{T}
    lie_basis::Vector{SparseMatrixCSC{T, Int}}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    diag_im_control_vec::Vector{T}
end

function precompute_derived_args(s::SystemParams)
    lie_basis = construct_lie_basis_general([s.im_control, s.im_drift])
    p_basis = lie_basis[2:end]
    diag_im_control_vec = Vector(diag(s.im_control))

    return DerivedArgs(lie_basis, p_basis, diag_im_control_vec)
end


struct PropagationParams{T}
    tmin::Float64
    tmax::Float64
    dt::Float64
    U0::Matrix{T}
    reg_coeff::Float64
    coset_tol::Float64
end

mutable struct StorageParams{T}
    adjoint_drift_tmp::Matrix{T}
    M_tmp::Matrix{T}
    U_tmp::Matrix{T}
    dU::Matrix{T}
    dM::Matrix{T}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
end 

struct Params{T}
    system_params::SystemParams{T}
    propagation_params::PropagationParams{T}
    storage_params::StorageParams{T}
    derived_args::DerivedArgs{T}
end 

function prepare_trivial_2D_setup()

    n_qubits = 2
    im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
    target = sparse(exp(-im*Matrix(construct_YQ_target(n_qubits))))
    system_params = SystemParams(im_drift, im_control, target)

    tmin = 0.1 * π
    tmax = 5 * π
    dt = (tmax - tmin) / 100
    dim = size(im_control, 1)
    U0 = Matrix{T}(I, dim, dim)
    coset_tol = 1e-6
    propagation_params = PropagationParams(tmin, tmax, dt, U0, 0.0, coset_tol)

    storage_params = StorageParams(
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim)
    )
    derived_args = precompute_derived_args(system_params)

    return Params(system_params, propagation_params, storage_params, derived_args)
end 