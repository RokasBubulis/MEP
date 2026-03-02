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

# set initial costate (fixed or random)
costate = zeros(ComplexF64, size(p_basis[1])...)
# m0 = [0.1, 0.4, 0.5, -0.5, -1.0, 0.2, 0.7]
m0 = rand(Float64, length(p_basis))
build_M!(costate, m0, p_basis)
println(m0)

mutable struct  Params{T}
    drift:: SparseMatrixCSC{T, Int} # -i*H0
    control:: SparseMatrixCSC{T, Int} # i*sum_j Z_j
    H_alpha_tmp::Matrix{T}
    min_alpha::Float64
    max_alpha::Float64
end

params = Params(-im*drift, im*control, zeros(ComplexF64, size(drift)), -Float64(pi/2), Float64(pi/2))

# optimise overlap 
optimal_adjoint_drift!(costate, params)

# plot overlap
α_grid = range(params.min_alpha, params.max_alpha, length=400)
vals = [-neg_adjoint_drift_obj([α], params.drift, params.control, costate) for α in α_grid]

optimal_overlap = real(tr(params.H_alpha_tmp * costate))
println(" optimal overlap: $optimal_overlap")
plot(α_grid, vals, label="objective")
hline!([optimal_overlap])

