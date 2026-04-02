using CMAEvolutionStrategy, Plots, Optim

include("propagation.jl")

function angles_to_directions(angles)
    n = length(angles) + 1
    m = zeros(n)
    m[1] = cos(angles[1])
    prefix = sin(angles[1])
    for i in 2:n-1
        m[i] = prefix * cos(angles[i])
        prefix *= sin(angles[i])
    end
    m[n] = prefix
    return m
end

##
params = prepare_trivial_2D_setup()
beta = 0.5
gamma = 0.0
bracket_order = 2

terms = String[]
beta != 0 && push!(terms, "$(beta)⋅drift")
gamma != 0 && push!(terms, "$(gamma)⋅p_basis[$bracket_order]")
target_str = "exp(-(" * join(terms, " + ") * "))"

params.system_params.target = sparse(exp(
    - beta * Matrix(params.system_params.im_drift) 
    - gamma * params.derived_args.p_basis[bracket_order]
))

# check target before propagation
check_unitarity(params.system_params.target, 0; note="Target")
target_dist = min_dist_to_target_coset(params.system_params.target, params)
@assert target_dist < params.propagation_params.coset_tol "dist(target) = $target_dist"

initial_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# initial_angles = [pi/4, pi/4, pi/4, pi/4, pi/4, pi/4]
objective(angles) = propagate_2nd_order(angles_to_directions(angles), params)

result = minimize(
    objective,
    initial_angles, 0.2;
    maxiter=50, verbosity=1
    )
angles_best = xbest(result)

# result = optimize(objective, initial_angles, Optim.NelderMead(), 
#                      Optim.Options(x_abstol=1e-14, f_abstol=1e-14))
# angles_best = Optim.minimizer(result)

m_best = angles_to_directions(angles_best)
println("m best: $m_best")

ts, Us, Ms, dists = propagate_2nd_order(m_best, params; store = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Lowest distance $(minimum(dists)) at time $(ts[argmin(dists)])")
remainder = Us[argmin(dists)] - params.system_params.target
# display(remainder)
# println(norm(remainder))

p1 = plot(ts, dists)
xlabel!(p1, "t")
ylabel!(p1, "d_targ_coset")

Ms_overlap_with_basis = [
    [abs(tr(Ms[i] / norm(Ms[i]) * (p/norm(p)))) for i in eachindex(ts)]
    for p in params.derived_args.p_basis
]
p2 = plot()
for (j, overlaps) in enumerate(Ms_overlap_with_basis)
    plot!(p2, ts, overlaps, label="basis $j")
end
xlabel!(p2, "t")
ylabel!(p2, "M(t) overlap with p basis elements")

mat = adjoint(params.system_params.target)
mat /= norm(mat)
Us_overlap_with_target = [abs(tr(Us[i] / norm(Us[1]) * mat)) for i in eachindex(ts)]
p3 = plot(ts, Us_overlap_with_target)
xlabel!(p3, "t")
ylabel!(p3, "U(t) overlap with target")

title_str = "min dist=$(round(min_dist, sigdigits=3)) at t=$(round(time_of_min_dist, sigdigits=3)), target=$target_str, \n M0=$([round(mi, sigdigits=2) for mi in m_best])"
plt = plot(p1, p2, p3, layout=(1,3), size=(1500, 500), plot_title=title_str, top_margin=20Plots.mm)
display(plt)