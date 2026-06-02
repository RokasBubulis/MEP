using DifferentialEquations, OrdinaryDiffEqLowOrderRK
using DiffEqCallbacks

include("adm.jl")
include("distance.jl")

function check_anti_hermiticity(H)
    # H = -adjoint(H)
    @assert isapprox(H, -adjoint(H), atol=1e-16, rtol=0) "Adjoint drift is not anti-hermitian"
end

function check_unitarity(U, tmp, tol)
    # U*adjoint(U) = I
    mul!(tmp, U, adjoint(U))
    nrm = norm(tmp) - sqrt(size(U,1))
    @assert nrm < tol "Propagator is not unitary: norm(U*adjoint(U) - I) = $nrm"
end

function f!(dx, x, p, t)
    algebra, stor, = p[1], p[4]
    optimal_adjoint_drift_lie!(stor.H_opt_lie, x, algebra, stor)
    lie_bracket_coeffs!(dx, algebra.structure_tensor, stor.H_opt_lie, x)
end

function propagate(m0::Vector{Float64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage, method; return_sol = false, kwargs...)

    # Initial values
    min_dist_tracker = MinDistanceTracker(1.0)
    m0_arr = zeros(Float64, length(algebra.lie_basis))
    m0_arr[2:end] = m0
    x0 = m0_arr

    # set up problem
    params = algebra, system, solver, stor, min_dist_tracker
    prob = ODEProblem(f!, x0, (0.0, solver.tmax), params)
    stor.U .= stor.U0

    # Step-wise computation to update minimum distance to target coset
    function step_update(u, t, integrator)
        system, solver, stor, tracker = integrator.p[2], integrator.p[3], integrator.p[4], integrator.p[5]
        optimal_adjoint_drift_lie!(stor.H_opt_lie, u, algebra, stor)
        Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
        check_anti_hermiticity(stor.H_opt)
        copyto!(stor.H_opt_dt, stor.H_opt)
        lmul!(integrator.dt, stor.H_opt_dt)
        stor.dU .= exp(stor.H_opt_dt)
        mul!(stor.U_buffer, stor.dU, stor.U)
        stor.U .= stor.U_buffer
        check_unitarity(stor.U, stor.U_unitary_buffer_check, solver.unitary_tol)

        dist = distance_objective_optimiser(stor.U, system, stor)
        if dist < 0.0
            @warn("Negative dist to target coset obtained: $dist, setting to positive")
            dist = abs(dist)
        end 
        tracker.min_dist = min(tracker.min_dist, dist)
    end

    # Functions to check if solution can be terminated
    function condition(u, t, integrator)
        return integrator.p[5].min_dist < integrator.p[3].dist_tol
    end

    function affect!(integrator)
        println("Distance $(integrator.p[5].min_dist) below tolerance $(integrator.p[3].dist_tol) reached at t = $(integrator.t). Terminating solver")
        terminate!(integrator)
    end

    # Build callbacks
    step_cb = FunctionCallingCallback(step_update; func_everystep=true)
    event_cb = DiscreteCallback(condition, affect!)
    cb = CallbackSet(step_cb, event_cb)

    # Solution 
    sol = solve(prob, method; callback=cb, reltol=solver.reltol, abstol=solver.abstol, kwargs...)

    if return_sol 
        return sol 
    else 
        return min_dist_tracker.min_dist
    end 
end 

#############

function find_best_initial_costate(algebra::Algebra, system::System, solver::SolverParams, stor::Storage, method=Tsit5(); show_trace=true, kwargs...)

    # check target before propagation
    check_unitarity(system.target, stor.tmp, solver.unitary_tol)
    targ_dist = distance(system.target, system, solver, stor)
    @assert targ_dist < solver.dist_tol "Error in target overlap: $targ_dist"

    n = length(algebra.p_basis)
    m0 = zeros(n)
    m0[1] = 1.0
    objective = function(m) 
        m ./= norm(m)
        propagate(m, algebra, system, solver, stor, method; save_everystep = false,
        save_start = false,
        save_end = false,
        dense = false, 
        kwargs...)
    end

    result = Optim.optimize(objective, m0, NelderMead(), Optim.Options(
        callback = state -> state.f_lowest < solver.dist_tol,
        show_trace  = show_trace, 
        g_abstol = solver.grad_tol,
        show_every=50
    ))
    m_best = result.minimizer
    dmin = result.minimum

    return m_best, dmin
end
