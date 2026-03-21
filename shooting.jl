using CMAEvolutionStrategy, Plots

include("propagation.jl")

params = prepare_trivial_2D_setup()
beta = 0.5
gamma = 0.0
params.system_params.target = sparse(exp(
    - beta * Matrix(params.system_params.im_drift) 
    - im *gamma * Matrix(construct_YQ_target(2))
    ))

# check target before propagation
check_unitarity(params.system_params.target, 0; note="Target")
target_dist = min_dist_to_target_coset(params.system_params.target, params)
@assert target_dist < params.propagation_params.coset_tol "dist(target) = $target_dist"

m0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = minimize((m)->propagate(m, params), m0, 0.5; maxiter=50, verbosity=1)
m_best = xbest(result)

ts, Us, Ms, Hs, dists = propagate_and_store_results(m_best, params)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]

println("Lowest distance $min_dist at time $time_of_min_dist")
p1 = plot(ts, dists, title="min dist=$(round(min_dist, sigdigits=3)) at t=$(round(time_of_min_dist, sigdigits=3))")
xlabel!(p1, "t")
ylabel!(p1, "d_targ_coset")

mat = params.system_params.im_drift
mat /= norm(mat)
Ms_overlap_with_drift = [abs(tr(Ms[i]/norm(Ms[i]) * mat)) for i in eachindex(ts)]
p2 = plot(ts, Ms_overlap_with_drift)
xlabel!(p2, "t")
ylabel!(p2, "M(t) overlap with drift")

mat = adjoint(params.system_params.target)
mat /= norm(mat)
Us_overlap_with_target = [abs(tr(Us[i] / norm(Us[1]) * mat)) for i in eachindex(ts)]
p3 = plot(ts, Us_overlap_with_target)
xlabel!(p3, "t")
ylabel!(p3, "U(t) overlap with U_T")

Ms_overlap_with_Hs = [abs(tr(Ms[i]/norm(Ms[i]) * Hs[i]/norm(Hs[i]))) for i in eachindex(ts)]
p4 = plot(ts, Ms_overlap_with_Hs)
xlabel!(p4, "t")
ylabel!(p4, "M(t) overlap with H_opt(t)")

plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
display(plt)