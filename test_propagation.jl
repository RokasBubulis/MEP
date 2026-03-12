using Plots

include("generators.jl")
include("lie_algebra.jl")
include("set_params.jl")
include("propagation.jl")

params = prepare_trivial_2D_setup()
p_basis = params.derived_args.p_basis
plt = plot(xlabel="Time (in π)", ylabel="Dist to target coset", grid=true)
n_runs = 20

for i in 1:n_runs
    local m0 = randn(length(p_basis))
    local ts, Ps, dists = propagate(m0, params)
    plot!(plt, ts, dists)
end

display(plt)
