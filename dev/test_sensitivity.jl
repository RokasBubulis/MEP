include("../src/structs.jl")
include("../src/sensitivity.jl")
include("../src/adm.jl")
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
stor = Storage(n_qubits, size(algebra.lie_basis, 1))
println("Setup finished")
##
# construct H_opt 
costate_arr = rand(8)
optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, costate_arr, algebra, stor)
b_dot_matrix!(algebra, stor)
d = copy(stor.db)
b_dot_elementwise!(algebra, stor)
a = copy(stor.db)

display(norm(a-d))

