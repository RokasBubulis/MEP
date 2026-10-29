include("../src/structs.jl")
include("../src/new_propagation.jl")
include("../src/optimise.jl")

using Plots, LaTeXStrings, Random

# same setup as test.jl
n_qubits = 2
tmax = 10.0
tstar_min = 0
tstar_max = tmax
phase = pi
gate_type = "k1"

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
tar = build_tar_ryd(gate_type, phase, n_qubits)
stor = Storage(n_qubits, size(algebra.lie_basis, 1))
println("Setup finished")

##
d = length(algebra.lie_basis)
Random.seed!(1)   # use the same seed in test.jl for an identical m0
m0 = rand(d)
T0 = 6.93
elapsed = @elapsed res = run_nelder_mead(m0, T0, algebra, solver, tar)

m, T = minimizer_mT(res, solver)
println("Nelder-Mead: J = ", Optim.minimum(res), ", T = ", T,
        ", iterations = ", Optim.iterations(res), ", shoots (f calls) = ", Optim.f_calls(res),
        ", converged = ", Optim.converged(res), ", time = ", round(elapsed, digits=1), " s")
