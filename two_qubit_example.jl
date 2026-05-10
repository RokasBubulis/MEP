include("structs.jl")
include("optimisation.jl")

using Plots


# target unitary by Lie coeffs
# lie_coeffs = zeros(8)
# lie_coeffs[1] = 0.0
# lie_coeffs[2] = 2.0
# # lie_coeffs[3] = 0.2
# lie_coeffs[6] = 0.3
# # lie_coeffs = [1.0, 1.0, 0.3, 
# #             0.4, 0.2, 0.6, 
# #             0.7, 0.1]

lie_coeffs = rand(Float64, 8) .*2 .-1
println(lie_coeffs)

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# solver parameters
tmax = 3.0
dt = 1e-2
tol = 1e-8
lambda = 0.0
Newton_steps = 50
Newton_tol = 1e-10
Newton_damping = 1.0
solver = SolverParams(tmax, dt, tol, lambda, Newton_steps, Newton_tol, Newton_damping)

# orthogonalise drift wrt control
c1 = tr(im_control * adjoint(im_drift))
c2 = tr(im_control * adjoint(im_control))
im_drift_orthogonal = im_drift - c1/c2 * im_control
@assert isapprox(tr(im_drift_orthogonal * adjoint(im_control)), 0.0, atol=1e-10)

# normalise generators 
im_control /= norm(im_control)
im_drift_orthogonal /= norm(im_drift_orthogonal)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift_orthogonal)
@assert length(lie_coeffs) == length(algebra.lie_basis)

# prepare system struct 
target = sparse(exp(-Matrix(
    sum(
        lie_coeffs[i] * algebra.lie_basis[i] for i in eachindex(lie_coeffs)
        )
    )))

# target = SparseMatrixCSC{ComplexF64, Int}(I, dim, dim)
# target[5,5] = -1.0

system = System{ComplexF64}(im_control, im_drift_orthogonal, target)

# prepare mutable storage 
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

# using Plots

# alphas = range(-20, 20, length=200)
# m0 = rand(7) .*2 .-1
# build_M0!(stor.M0, m0, algebra)
# println(m0)

# obj_l2_values = [adjoint_drift_obj(α, stor.M0, algebra, solver, stor) for α in alphas]
# obj_values = [adjoint_drift_obj(α, stor.M0, algebra, solver, stor) + solver.lambda * α^2 for α in alphas]

# p = plot(alphas, obj_l2_values,
#     xlabel = "α",
#     label = "L2 objective"
# )
# plot!(alphas, obj_values, label="Real objective")

# display(p)

# using Plots
# using Peaks  # for findmaxima/findminima

# alphas = range(-20, 20, length=200)
# m0 = rand(7) .*2 .-1
# build_M0!(stor.M0, m0, algebra)
# println(m0)

# obj_l2_values = [adjoint_drift_obj(α, stor.M0, algebra, solver, stor) for α in alphas]
# obj_values = [adjoint_drift_obj(α, stor.M0, algebra, solver, stor) + solver.lambda * α^2 for α in alphas]

# p = plot(alphas, obj_l2_values, xlabel="α", label="L2 objective")
# plot!(alphas, obj_values, label="Real objective")

# # annotate maxima and minima for obj_values
# max_idxs, _ = findmaxima(obj_values)
# min_idxs, _ = findminima(obj_values)

# scatter!(alphas[max_idxs], obj_values[max_idxs], 
#     series_annotations=text.(round.(alphas[max_idxs], digits=2), :bottom, 8),
#     markershape=:utriangle, label="maxima")
# scatter!(alphas[min_idxs], obj_values[min_idxs], 
#     series_annotations=text.(round.(alphas[min_idxs], digits=2), :top, 8),
#     markershape=:dtriangle, label="minima")

# display(p)

# results
m_best = find_best_initial_costate_autograd(algebra, system, solver, stor)
ts, Us, Ms, dists = propagate(m_best, algebra, system, solver, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Target in Lie coeffs: $lie_coeffs")
println("Autograd Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)

# # p = plot(ts, dists)
# # display(p)

# alphas = range(-100, 100, 1000)
# errors_Campbell_br = zeros(length(alphas))
# errors_Campbell_tensor = zeros(length(alphas))
# errors_series = zeros(length(alphas))
# errors_true = zeros(length(alphas))
# remainders_br = zeros(length(alphas))
# remainders_tensor = zeros(length(alphas))
# for (i,α) in enumerate(alphas)
#     adjoint_action_by_campbell!(stor.M0, -α * system.im_control, -system.im_drift, stor)
#     stor.adjoint_drift = adjoint_action_true(-α * system.im_control, -system.im_drift)
#     errors_Campbell_br[i] = error_belongs_to_p_subspace(stor.M0, algebra)
#     remainders_br[i] = norm(stor.M0 - stor.adjoint_drift)
#     adjoint_action_by_campbell_structure_tensor!(stor.M0, -α * system.im_control, -system.im_drift, algebra, stor)
#     errors_Campbell_tensor[i] = error_belongs_to_p_subspace(stor.M0, algebra)
#     remainders_tensor[i] = norm(stor.M0 - stor.adjoint_drift)
#     # adjoint_action_series!(stor.tmp, -α * system.im_control, -system.im_drift, stor)
#     # errors_series[i] = error_belongs_to_p_subspace(stor.tmp, algebra)
#     # stor.tmp = adjoint_action_true(-α * system.im_control, -system.im_drift)
#     # errors_true[i] = error_belongs_to_p_subspace(stor.tmp, algebra)
# end 
# p = plot()
# p = plot(alphas, errors_Campbell_br, label="Campbell bracket error", yscale=:log10)
# plot!(alphas, remainders_br, label="Remainder Campbell Bracket - True", yscale=:log10)
# plot!(p, alphas, remainders_tensor, label="Remainder Campbell Tensor - True", yscale=:log10)
# plot!(p, alphas, errors_Campbell_tensor, label="Campbell tensor error", yscale=:log10)
# # plot!(p, alphas, errors_series, label="Series", yscale=:log10)
# # plot!(p, alphas, errors_true, label="True", yscale=:log10)
# xlabel!("α")
# ylabel!("norm(residual) out of p-basis")
# display(p)
