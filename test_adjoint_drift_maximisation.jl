using Plots
using Random

include("generators.jl")
include("lie_algebra.jl")
include("adjoint_drift_maximisation.jl")

T = float_type

function build_M!(M::AbstractMatrix{T}, m::AbstractVector{Float64}, p_basis::Vector{SparseMatrixCSC{T, Int}})
    fill!(M, zero(T))
    for (i, m_coeff) in enumerate(m)
        M .+= m_coeff * p_basis[i]
    end
    return nothing
end 

n_qubits = 2
control, drift = construct_Ryd_generators(n_qubits)
gens = [control, drift]
lie_basis = construct_lie_basis_general(gens)
p_basis = lie_basis[2:end]
M = zeros(ComplexF64, size(p_basis[1])...)
m0 = [0.1, 0.4, 0.5, -0.5, -1.0, 0.2, 0.7]
build_M!(M, randn(length(p_basis)), p_basis)
#build_M!(M, m0, p_basis)

mutable struct  Params{T}
    drift:: SparseMatrixCSC{T, Int} # -i*H0
    control:: SparseMatrixCSC{T, Int} # -i*sum_j Z_j
    H_alpha_tmp::Matrix{T}
end

params = Params(-im*drift, im*control, zeros(ComplexF64, size(drift)))

adjoint_action_by_campbell(0.5 * params.control, params.drift; depth = 5)

display(params.H_alpha_tmp)
optimal_adjoint_drift!(M, params)
display(params.H_alpha_tmp)
display(params.drift - params.H_alpha_tmp)

# α_grid = range(-π, π, length=400)
# vals = [-neg_adjoint_drift_obj([α], params.drift, params.control, M) for α in α_grid]

# optimal_overlap = real(tr(params.H_alpha_tmp * M))
# println(" optimal overlap: $optimal_overlap")
# plot(α_grid, vals, label="objective")
# hline!([optimal_overlap])

