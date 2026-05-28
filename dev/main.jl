include("../src/structs.jl")
include("../src/propagation.jl")

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# prepare system struct 
target = Matrix(SparseMatrixCSC{ComplexF64, Int}(I, dim, dim))
target[5,5] = -1.0
title="CZ"
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
system = System{ComplexF64}(im_control, im_drift, target, target_logic)
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))
##

# m0 = rand(length(algebra.p_basis))
m0 = zeros(length(algebra.p_basis))
m0[1] = 1.0
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
opt_tol = 1e-8
method=Midpoint()
dt = 1e-2

solver = SolverParams(tmax, reltol, abstol, dist_tol, opt_tol)
m_best, dmin = find_best_initial_costate(algebra, system, solver, stor, method, dt=dt, adaptive=false)
sol = propagate(m_best, algebra, system, solver, stor, method, dt=dt, saveat=solver.tmax, return_sol=true)
Us = [u.U for u in sol.u]
dists = [distance(Us[i], system, solver, stor) for i in eachindex(sol.t)]
println("Min dist of $dmin at t=$(sol.t[argmin(dists)])")

# smth off due to negative min dist