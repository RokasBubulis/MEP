include("generators.jl")
include("lie_algebra.jl")
T = ComplexF64

# set to mutable for target = exp(-drift) testing
mutable struct SystemParams{T}
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
    lie_basis = construct_lie_basis_general([copy(s.im_control), copy(s.im_drift)])
    p_basis = lie_basis[2:end]
    diag_im_control_vec = Vector(diag(copy(s.im_control)))

    return DerivedArgs(lie_basis, p_basis, diag_im_control_vec)
end

struct PropagationParams{T}
    tmax::Float64
    dt::Float64
    U0::Matrix{T}
    coset_tol::Float64
end

mutable struct StorageParams{T}
    adjoint_drift_tmp::Matrix{T}
    M_tmp0::Matrix{T}
    M_tmp1::Matrix{T}
    M_tmp2::Matrix{T}
    U_tmp::Matrix{T}
    dU::Matrix{T}
    dM::Matrix{T}
    tmp_mat::Matrix{T}
end 

struct Params{T}
    system_params::SystemParams{T}
    propagation_params::PropagationParams{T}
    derived_args::DerivedArgs{T}
end 

function prepare_trivial_2D_setup()

    n_qubits = 2
    im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
    dim = size(im_control, 1)
    target = SparseMatrixCSC{T, Int}(undef, dim, dim)
    system_params = SystemParams(im_drift, im_control, target)

    tmax = 1.0
    dt = tmax / 100
    dim = size(im_control, 1)
    U0 = Matrix{T}(I, dim, dim)
    coset_tol = 1e-5
    propagation_params = PropagationParams(tmax, dt, U0, coset_tol)

    storage_params = StorageParams(
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim)
    )
    derived_args = precompute_derived_args(system_params)

    return Params(system_params, propagation_params, derived_args), storage_params
end 
