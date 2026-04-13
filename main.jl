using Optim, CMAEvolutionStrategy

include("params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")
include("propagation.jl")

# specify target over lie basis
lie_coeffs = [0.1, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
tmax = 1.0
dt = tmax / 100
tol = 1e-5

params, stor = prepare_2q_setup_with_target_from_Lie_coeffs(lie_coeffs, tmax, dt, tol)

m_best = find_best_initial_costate(params, stor)
ts, Us, Ms, Hs, dists = propagate_2nd_order(m_best, params, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Lowest distance $(minimum(dists)) at time $(ts[argmin(dists)])")