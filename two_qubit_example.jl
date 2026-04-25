include("structs.jl")
include("optimisation.jl")


# target unitary by Lie coeffs
lie_coeffs = zeros(8)
lie_coeffs[1] = 0.6
lie_coeffs[2] = 0.6
lie_coeffs[3] = 0.2

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# solver parameters
tmax = 1.0
dt = tmax / 100
tol = 1e-8
lambda = 0.0
Newton_steps = 100
Newton_tol = 1e-10
solver = SolverParams(tmax, dt, tol, lambda, Newton_steps, Newton_tol)

# orthogonalise drift wrt control
c1 = tr(im_control * adjoint(im_drift))
c2 = tr(im_control * adjoint(im_control))
im_drift_orthogonal = im_drift - c1/c2 * im_control
@assert isapprox(tr(im_drift_orthogonal * adjoint(im_control)), 0.0, atol=1e-10)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift_orthogonal)
@assert length(lie_coeffs) == length(algebra.lie_basis)

# prepare system struct 
target = sparse(exp(-Matrix(
    sum(
        lie_coeffs[i] * algebra.lie_basis[i] for i in eachindex(lie_coeffs)
        )
    )))
system = System{ComplexF64}(im_control, im_drift_orthogonal, target)

# prepare mutable storage 
stor = Storage{ComplexF64}(dim)

# results
m_best = find_best_initial_costate_autograd(algebra, system, solver, stor)
ts, Us, Ms, dists = propagate_2nd_order(m_best, algebra, system, solver, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Target in Lie coeffs: $lie_coeffs")
println("Autograd Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)


