include("propagation.jl")

params = prepare_trivial_2D_setup()

coeffs = [0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]

terms = String[]
for (i, coeff) in enumerate(coeffs)
    coeff != 0 && push!(terms, "$(coeff)⋅p_[$i]")
end 
target_str = "exp(-(" * join(terms, " + ") * "))"

params.system_params.target = sparse(exp(
    - Matrix(sum(coeffs[i] * params.derived_args.p_basis[i] for i in eachindex(coeffs)))
))

m_best = find_best_initial_costate(params)
ts, Us, Ms, Hs, dists = propagate_2nd_order(m_best, params; store = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Lowest distance $(minimum(dists)) at time $(ts[argmin(dists)])")