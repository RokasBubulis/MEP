include("structs.jl")
include("optimisation.jl")

using Plots

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)
##

# solver parameters
tmax = 10
dt = 5e-2
tol = 1e-8
Newton_steps = 50
Newton_tol = 1e-10
Newton_damping = 1.0
solver = SolverParams(tmax, dt, tol, Newton_steps, Newton_tol, Newton_damping)

# prepare system struct 
target = Matrix(SparseMatrixCSC{ComplexF64, Int}(I, dim, dim))
target[5,5] = -1.0
title="CZ"
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
system = System{ComplexF64}(im_control, im_drift, target, target_logic)

# prepare mutable storage 
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

# results
#m_best = find_best_initial_costate_autograd(algebra, system, solver, stor)
m_best = find_best_initial_costate_bbf(algebra, system, solver, stor)
ts, Us, Ms, dists = propagate_RK4(m_best, algebra, system, solver, stor; save = true)
min_dist = minimum(dists)
time_of_min_dist = ts[argmin(dists)]
println("Lowest distance $min_dist at time $(ts[argmin(dists)])")
println(m_best)

p = plot(ts, dists)
title!(p, title * "\n tmax: $tmax, dt: $dt, dmin: $(round(min_dist, sigdigits=3)), t*: $(round(time_of_min_dist, sigdigits=4))")
display(p)