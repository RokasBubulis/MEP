include("../src/structs.jl")
include("../src/new_propagation.jl")
include("../src/optimise.jl")

using Kronecker, Plots, LaTeXStrings, Random, Printf

# generators (test_forward.jl setup; the T window is widened so that T0 = 5 is an interior initial guess)
n_qubits = 2
tmax = 20.0
tstar_min = 3.0      # T is optimised in (tstar_min, tstar_max)
tstar_max = tmax
phase = 0.1*pi
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
# multi-start BFGS on J(m, T) = φ(U(T; m)) + time_weight * T: several random costate directions m0,
# all starting from the same initial guess T0 for the final time
d = length(algebra.lie_basis)                      # initial guess for T
n_starts = 10                    # all starts are run: gates exist at several T, the shortest one is wanted
time_weight = 5e-3
nsteps = round(Int, tmax/dt)     # dt = T/nsteps ≤ 1e-2 for every T in the window

@time begin
    best = (cost = Inf, J = Inf, m = zeros(d), T = 0.0)
    for k in 1:n_starts
        m0_k = randn(d)
        T0 = tstar_min + rand(1)[] * (tmax - tstar_min)          
        res_k = run_bfgs(m0_k, T0, algebra, solver, tar; time_weight=time_weight, nsteps=nsteps)
        m_k, T_k = minimizer_mT(res_k, solver)
        cost_k = Optim.minimum(res_k) - time_weight*T_k
        @printf("start %2d: cost = %.3e at T = %.4f (%d iterations)\n", k, cost_k, T_k, Optim.iterations(res_k))
        # keep the start with the lowest J = φ + time_weight*T, so a shorter gate beats a longer one
        if Optim.minimum(res_k) < best.J
            global best = (cost = cost_k, J = Optim.minimum(res_k), m = m_k, T = T_k)
        end
    end

    # polish the best start without the time penalty (J = φ), so T settles at the gate time itself
    res = run_bfgs(best.m, best.T, algebra, solver, tar; time_weight=0.0, nsteps=nsteps, iterations=300)
    m_best, T_best = minimizer_mT(res, solver)
    tracker = new_propagate(m_best, algebra, solver, tar; T=T_best, nsteps=nsteps, track_min=false) 
    println("Final: cost = ", tracker.min_cost, " at T = ", T_best, " (polish: ", Optim.iterations(res), " iterations, converged = ", Optim.converged(res), ")")
end