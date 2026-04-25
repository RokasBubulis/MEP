using Optim, Plots

include("params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")
include("propagation.jl")

##
tmax = 1.0
dt = tmax / 100
tol = 1e-8
lambda = 0.0
Newton_steps = 100
Newton_tol = 1e-10

lie_coeffs = zeros(8)
lie_coeffs[1] = 0.3
lie_coeffs[2] = 0.4

lie_coeffs[3] = 0.8
#lie_coeffs[5] = 0.3

im_control, im_drift = im .* construct_Ryd_generators(2)
c1 = tr(im_control * adjoint(im_drift))
c2 = tr(im_control * adjoint(im_control))
im_drift_orthogonalised = im_drift - c1/c2 * im_control

algebra = make_algebra(im_control, im_drift_orthogonalised)

drift_lie = project_algebra(im_drift_orthogonalised, algebra)

control_lie = project_algebra(im_control, algebra)

lie_coeffs .+= 0.4 * drift_lie[:] + 0.2 * control_lie[:]

params = prepare_2q_setup_with_target_from_Lie_coeffs(lie_coeffs, im_drift_orthogonalised, im_control, algebra, tmax, dt, tol, lambda, Newton_steps, Newton_tol)
# params = prepare_2q_setup_CZ(tmax, dt, tol, lambda, Newton_steps, Newton_tol)
dim = size(params.physics.im_control, 1)
stor = StorageParams{ComplexF64}(dim)


m_best = find_best_initial_costate_autograd(params, stor)
ts, Us, Ms, dists = propagate_2nd_order(m_best, params, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Target in Lie coeffs: $lie_coeffs")
println("Autograd Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)
p = plot(ts, dists, yscale=:log10, label="Autograd")
display(p)
