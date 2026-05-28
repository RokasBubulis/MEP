using Plots, Random, BenchmarkTools

include("adjoint_drift_maximisation.jl")
include("set_params.jl")
include("propagation.jl")

params = prepare_trivial_2D_setup()
min_alpha = -10
max_alpha = 10

p_basis = params.derived_args.p_basis
costate = zeros(T, size(p_basis[1])...)
m0 = rand(Float64, length(p_basis))
build_M0!(costate, m0, params)

# optimise overlap
optimal_adjoint_drift_newton!(costate, params)
# plot overlap
α_grid = range(min_alpha, max_alpha, length=1000)
vals = [-neg_adjoint_drift_obj([α], 
    -params.derived_args.p_basis[1], 
    params.system_params.im_control, 
    costate)
    for α in α_grid]

optimal_overlap = real(tr(params.storage_params.adjoint_drift_tmp * costate))
println(" optimal overlap: $optimal_overlap")
plot(α_grid, vals, label="objective")
hline!([optimal_overlap])
