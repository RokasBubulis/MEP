using Plots, Random, BenchmarkTools

include("adjoint_drift_maximisation.jl")
include("set_params.jl")

function build_M!(M::AbstractMatrix{T}, m::AbstractVector{Float64}, p_basis::Vector{SparseMatrixCSC{T, Int}})
    fill!(M, zero(T))
    for (i, m_coeff) in enumerate(m)
        M .+= m_coeff * p_basis[i]
    end
    M /= norm(M)
    return nothing
end 

params = prepare_trivial_2D_setup()
min_alpha = -30
max_alpha = 30

p_basis = params.derived_args.p_basis
costate = zeros(T, size(p_basis[1])...)
m0 = rand(Float64, length(p_basis))
build_M!(costate, m0, p_basis)

# optimise overlap
optimal_adjoint_drift_newton!(costate, params)
# plot overlap
α_grid = range(min_alpha, max_alpha, length=1000)
vals = [-neg_adjoint_drift_obj([α], 
    -params.system_params.im_drift, 
    params.system_params.im_control, 
    costate, 
    params.propagation_params.reg_coeff)
    for α in α_grid]

optimal_overlap = real(tr(params.storage_params.adjoint_drift_tmp * costate))
println(" optimal overlap: $optimal_overlap")
plot(α_grid, vals, label="objective")
hline!([optimal_overlap])
