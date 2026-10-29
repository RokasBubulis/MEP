include("../src/structs.jl")
include("../src/new_propagation.jl")
include("../src/optimise.jl")

using Kronecker, Plots, LaTeXStrings, Random

# generators (same setup as test_forward.jl)
n_qubits = 2
tmax = 7.5
tstar_min = 2.0
tstar_max = tmax
phase = pi
gate_type = "k1"
dt=1e-2

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
#tar = build_tar_ryd(gate_type, phase, n_qubits)
tar = build_tar(gate_type, phase, n_qubits, algebra)
println("Setup finished")
##
# gradient based optimisation (BFGS) of the smooth objective J(m, T) = φ(U(T; m)) + time_weight * T
# over the costate direction m and the final time T ∈ (tstar_min, tstar_max)
d = length(algebra.lie_basis)
# Random.seed!(3)
m0 = rand(d)
T0 = 5.0
nsteps = round(Int, T0/dt)     # dt = T/nsteps ≈ 1e-2 as in test_forward.jl
res = run_bfgs(m0, T0, algebra, solver, tar; nsteps=nsteps)

m_best, T_best = minimizer_mT(res, solver)
tracker = new_propagate(m_best, algebra, solver, tar; T=T_best, nsteps=nsteps, track_min=false)
println("Final: cost = ", tracker.min_cost, " at T = ", T_best, " after ", Optim.iterations(res), " iterations (converged = ", Optim.converged(res), ")")
