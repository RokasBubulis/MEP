include("../src/structs.jl")
include("../src/new_propagation.jl")
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
tar = build_tar_ryd(gate_type, phase, n_qubits)
#tar = build_tar(gate_type, phase, n_qubits, algebra)
stor = Storage(n_qubits, size(algebra.lie_basis, 1))
println("Setup finished")
##

m0 = rand(7)
tracker = new_propagate(m0, algebra, solver, tar)
analytic_grad = zeros(Float64, 8)
grad_infidelity!(analytic_grad, algebra, tracker)
##
dm = 1e-3
finite_diff_grad = zeros(Float64, 7)
for i in 1:7
    mp = copy(m0); mp[i] += dm 
    mm = copy(m0); mm[i] -= dm 
    tracker_p = new_propagate(mp, algebra, solver, tar)
    tracker_m = new_propagate(mm, algebra, solver, tar)
    cost_p = tracker_p.min_cost
    cost_m = tracker_m.min_cost
    finite_diff_grad[i] = (cost_p - cost_m) / (2*dm)
    println(tracker_m.t_star == tracker_p.t_star)
end 
##
display(analytic_grad)
display(finite_diff_grad)

# time threshold for cost optimum update pushes finite diff grad from zeros but the consequence is that the time at which cost is evaluated is not the same for + and - runs. Also worth investigating is whether the time matches with analytic run and why analytic grad produces a non zero first entry