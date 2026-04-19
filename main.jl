using Optim, CMAEvolutionStrategy, Plots

include("params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")
include("propagation.jl")

tmax = 1.0
dt = tmax / 100
tol = 1e-8
lambda = 0.0
Newton_steps = 50

# specify target over lie basis
lie_coeffs = [1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
#lie_coeffs = rand(8) * 2 .-1 ./2
#println(lie_coeffs)
params, stor = prepare_2q_setup_with_target_from_Lie_coeffs(lie_coeffs, tmax, dt, tol, lambda, Newton_steps)

#params, stor = prepare_2q_setup_CZ(tmax, dt, tol)

m_best = find_best_initial_costate_autograd(params, stor)
ts, Us, Ms, Hs, dists = propagate_2nd_order(m_best, params, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)
p = plot(ts, dists, yscale=:log10)
xlabel!(p, "t")
ylabel!(p, "Min dist")
display(p)

# elements = params.algebra.p_basis
# p1 = plot()
# for j in eachindex(elements)
#     plot!(p1, ts[1:argmin(dists)], [real(tr(Ms[i]*elements[j])) for i in 1:argmin(dists)])
# end 

# display(plot(p, p1, layout=(2,1)))

# function distance_plot(U::Union{Matrix{T}, SparseMatrixCSC{T,Int}}, 
#     params::Params, stor::StorageParams)

#     mul!(stor.tmp, U, params.physics.adjoint_target)
#     tmp_diag = diag(stor.tmp)

#     dist(β) = 1 - 1/size(U,1) * abs(dot(exp.(β.*params.physics.im_control_vec), tmp_diag))
#     betas = collect(range(-pi, pi, 100))
#     display(plot(betas, [dist(beta) for beta in betas]))
#     res = optimize(dist, -pi, pi)
#     β_opt = Optim.minimizer(res)
#     println(β_opt)
#     return dist(β_opt)
# end