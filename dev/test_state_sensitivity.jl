include("../src/structs.jl")
include("../src/new_propagation.jl")

# generators
n_qubits = 2
tmax = 2.0
tstar_min = 0
tstar_max = tmax
phase = pi
gate_type = "k1"
dt=1e-3

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
tracker = new_propagate(m0, algebra, solver, tar)

dm = 1e-3
for μ in 1:d 
    dU_dm_mu_analytic = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν] for ν in 1:d) * tracker.U_t_star
    mp = copy(m0); mp[μ] += dm 
    mm = copy(m0); mm[μ] -= dm 
    tracker_mp = new_propagate(mp, algebra, solver, tar)
    tracker_mm = new_propagate(mm, algebra, solver, tar)
    dU_dm_mu_finite_diff = (tracker_mp.U_t_star- tracker_mm.U_t_star) / (2*dm)
    println(norm(dU_dm_mu_analytic - dU_dm_mu_finite_diff))
end 
