include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, Plots, LaTeXStrings

# generators
n_qubits = 2
tmax = 7.5
tstar_min = 5
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
find_best_initial_costate(tar, algebra, solver; dt=dt, show_trace=false)
