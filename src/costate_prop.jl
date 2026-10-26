using DifferentialEquations, DiffEqCallbacks, OrdinaryDiffEqLowOrderRK, StaticArrays, RecursiveArrayTools, LinearAlgebra

# Costate-only subset of new_propagation.jl:
#   M   : adjoint/costate variables, propagated by the solver   (dM/dt = [H_opt, M])
#   chi : costate sensitivity, propagated by the solver         (dchi/dt = chi D)
#   U   : unitary, propagated manually in the callback (needed for H_opt and cost)
# State-sensitivity (E, F, B) propagation is deliberately omitted.

function costate_ode_problem!(du, u, p, t)
    M, χ = u.x
    dM, dχ = du.x

    algebra, stor = p[1], p[2]

    # costate propagation
    # obtain H_opt
    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, M, algebra, stor)
    # dM/dt = [H_opt, M] in Lie space; dim = dx1
    lie_bracket_coeffs!(dM, algebra.structure_tensor, stor.H_opt_lie, M)

    # costate sensitivity propagation
    # dχ/dt = χD in Lie space; dim = dxd
    costate_sensitivity_drift!(stor.D, algebra, stor, M)
    mul!(dχ, χ, stor.D)
    nothing
end

function costate_unconditional_optimum_update!(u, t, algebra::Algebra, stor::Storage, tar::TargetContainer, tracker::CostOptimumTracker)
    cost = infidelity(stor.U, algebra, stor, tar)
    M, χ = u.x
    tracker.min_cost = cost
    tracker.t_star = t
    tracker.U_t_star .= stor.U
    tracker.W_t_star .= conj(exp(stor.β_opt * Matrix(algebra.lie_basis[1])) * tar.target)
    tracker.M_t_star .= M
    tracker.χ_t_star .= χ
end

function costate_propagate(m0::Vector{Float64}, algebra::Algebra, solver::SolverParams, tar::TargetContainer; dt=1e-2, method=Midpoint())

    # initialisation
    d = length(algebra.lie_basis)
    n = algebra.n_levels^algebra.n_particles
    stor = Storage(algebra.n_particles, d)

    x0 = ArrayPartition(m0, Matrix{Float64}(I, d, d))

    # setup optimal point tracker
    tracker = CostOptimumTracker(1.0, 0.0, Matrix{Float64}(undef, d, d), Matrix{ComplexF64}(undef, n, n), Matrix{ComplexF64}(undef, n, n), Vector{Float64}(undef, d), Matrix{Float64}(undef, d, d))

    # setup ODE problem
    params = algebra, stor, tracker, tar, solver
    prob = ODEProblem(costate_ode_problem!, x0, (0.0, solver.tmax), params)
    # solve
    sol = solve(prob, method; reltol=solver.reltol, abstol=solver.abstol, adaptive=false, dt=dt, save_everystep=false, dense=false)

    costate_unconditional_optimum_update!(sol.u[end], sol.t[end], algebra, stor, tar, tracker)

    return tracker
end
