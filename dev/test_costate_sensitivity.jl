include("../src/structs.jl")
include("../src/new_propagation.jl")
include("../src/costate_prop.jl")

# generators
n_qubits = 2
tmax = 2.0
tstar_min = 0
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
d = length(algebra.lie_basis)
m0 = rand(d)
tracker = costate_propagate(m0, algebra, solver, tar)

dm = 1e-4
for μ in 1:d 
    dM_mu_analytic = tracker.χ_t_star[μ, :]
    mp = copy(m0); mp[μ] += dm 
    mm = copy(m0); mm[μ] -= dm 
    tracker_mp = costate_propagate(mp, algebra, solver, tar)
    tracker_mm = costate_propagate(mm, algebra, solver, tar)
    dM_mu_finite_diff = (tracker_mp.M_t_star- tracker_mm.M_t_star) / (2*dm)
    println(norm(dM_mu_analytic - dM_mu_finite_diff))
end 
