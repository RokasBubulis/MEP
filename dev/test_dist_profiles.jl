include("structs.jl")
include("optimisation.jl")

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# solver parameters
tmax = 10
dt = 1e-3
tol = 1e-10
Newton_steps = 50
Newton_tol = 1e-10
Newton_damping = 1.0

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
solver = SolverParams(tmax, dt, tol, Newton_steps, Newton_tol, Newton_damping)

m0= rand(length(algebra.p_basis))
ts, Us_mp, Ms_mp, dists_mp = propagate_MP(m0, algebra, system, solver, stor; save = true)
ts, Us_rk, Ms_rk, dists_rk = propagate_RK4(m0, algebra, system, solver, stor; save = true)
p = plot(ts, dists_mp, label="Midpoint")
plot!(p, ts, dists_rk, label="RK4")
xlabel!(p, "time")
ylabel!(p, "distance")
title!("m0 = $(round.(m0, sigdigits=2)), \n dt = $(solver.dt)")
savefig(p, "results/dist_vs_t_comparison_dt_$(solver.dt).png")
display(p)