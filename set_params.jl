include("generators.jl")
include("lie_algebra.jl")
T = ComplexF64

struct SystemParams{T}
    im_drift::SparseMatrixCSC{T, Int} # i H_0
    im_control::SparseMatrixCSC{T, Int}  # i sum_j Z_j
    target::SparseMatrixCSC{T, Int}  # U_T 
end

lie_basis(s::SystemParams) = construct_lie_basis_general([s.im_control, s.im_drift])
orth_control_complement_basis(s::SystemParams) = lie_basis(s)[2:end]
diagonal_control_vec(s::SystemParams) = Vector(diag(s.im_control))

struct PropagationParams{T}
    tmin::Float64
    tmax::Float64
    dt::Float64
    P0::Matrix{T}
    reg_coeff::Float64
end

mutable struct StorageParams{T}
    H_alpha_tmp::Matrix{T}
    M_tmp::Matrix{T}
    P_tmp::Matrix{T}
    dP::Matrix{T}
    dM::Matrix{T}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
end 

struct Params{T}
    system_params::SystemParams{T}
    propagation_params::PropagationParams{T}
    storage_params::StorageParams{T}
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
    P0 = Matrix{T}(I, dim, dim)
    propagation_params = PropagationParams(tmin, tmax, dt, P0, 0.0)

    storage_params = StorageParams(
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim)
    )
    return Params(system_params, propagation_params, storage_params)
end 