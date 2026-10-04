include("../src/structs.jl")
include("../src/backpropagation.jl")
using Kronecker, Plots, LaTeXStrings

# generators
n_qubits = 2
tmax = 20
tstar_min = 5
tstar_max = tmax
phase = pi / 2
gate_type = "k0"
dt=1e-2

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
tar = build_tar_ryd(gate_type, phase, n_qubits)
println("Setup finished")
##
find_best_final_beta(tar, algebra, solver; dt=dt, show_trace=false)