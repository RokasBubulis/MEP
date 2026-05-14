include("structs.jl")
include("optimisation.jl")

using Plots


# target unitary by Lie coeffs
lie_coeffs = zeros(8)
lie_coeffs[1] = 0.0
lie_coeffs[2] = 2.0
# lie_coeffs[3] = 0.2
lie_coeffs[6] = 0.4
# # lie_coeffs = [1.0, 1.0, 0.3, 
# #             0.4, 0.2, 0.6, 
# #             0.7, 0.1]

# lie_coeffs = rand(Float64, 8) .*2 .-1
# println(lie_coeffs)

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

# target = Matrix(SparseMatrixCSC{ComplexF64, Int}(I, dim, dim))
# target[5,5] = -1.0

system = System{ComplexF64}(im_control, im_drift_orthogonal, target)

# prepare mutable storage 
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

# results
m_best = find_best_initial_costate_autograd(algebra, system, solver, stor)
#m_best = find_best_initial_costate_bbf(algebra, system, solver, stor)
ts, Us, Ms, dists = propagate(m_best, algebra, system, solver, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Target in Lie coeffs: $lie_coeffs")
println("Autograd Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)

# p = plot(ts, dists)
# display(p)