using DifferentialEquations, OrdinaryDiffEqLowOrderRK
using DiffEqCallbacks

include("adm.jl")
include("distance.jl")

# if used should be in place
# function check_anti_hermiticity(H)
#     # H = -adjoint(H)
#     @assert isapprox(H, -adjoint(H), atol=1e-16, rtol=0) "Adjoint drift is not anti-hermitian"
# end

function check_unitarity(U, U_adj, tmp, tol)
    # U*adjoint(U) = I
    adjoint!(U_adj, U)
    mul!(tmp, U, U_adj)
    @inbounds for i in 1:size(tmp, 1)
        tmp[i, i] -= one(eltype(tmp))
    end
    
    nrm = norm(tmp)
    @assert nrm < tol "Propagator is not unitary: norm(U*adjoint(U) - I) = $nrm"
end

# propagation of costate: dM/dt = [H_opt(t), M(t)]
function f!(dx, x, p, t)
    algebra, stor, = p[1], p[4]
    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, x, algebra, stor)
    lie_bracket_coeffs!(dx, algebra.structure_tensor, stor.H_opt_lie, x)
end

# Step-wise computation to update minimum distance to target coset
function propagate_U_and_update_distance!(u, t, integrator)
    tar, solver, stor, tracker = integrator.p[2], integrator.p[3], integrator.p[4], integrator.p[5]

    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, u, algebra, stor)
    Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
    #check_anti_hermiticity(stor.H_opt)
    copyto!(stor.dU, stor.H_opt)
    lmul!(integrator.dt, stor.dU)
    exponential!(stor.dU, stor.exp_method, stor.exp_cache)
    mul!(stor.U_buffer, stor.dU, stor.U)
    stor.U .= stor.U_buffer
    check_unitarity(stor.U, stor.U_adj_buffer, stor.U_unitary_buffer_check, solver.unitary_tol)

    dist = distance(stor.U, tar, algebra, stor)
    if dist < 0.0
        @warn("Negative dist to target coset obtained: $dist, setting to positive")
        dist = abs(dist)
    end 
    if dist < tracker.min_dist
        tracker.min_dist = dist 
        tracker.tstar = t 
    end 
end

function propagate(m0::Vector{Float64}, algebra::Algebra, solver::SolverParams, stor::Storage, tar::TargetContainer, method; full_results = false, kwargs...)

    # Initial values
    tracker = OutputTracker(1.0, solver.tmax)
    m0_arr = zeros(Float64, length(algebra.lie_basis))
    m0_arr[2:end] = m0
    x0 = m0_arr

    # set up problem
    params = algebra, tar, solver, stor, tracker
    prob = ODEProblem(f!, x0, (0.0, solver.tmax), params)
    stor.U .= stor.U0

    # Functions to check if solution can be terminated
    function condition(u, t, integrator)
        return integrator.p[5].min_dist < integrator.p[3].dist_tol
    end

    function affect!(integrator)
        #println("Distance $(integrator.p[5].min_dist) below tolerance $(integrator.p[3].dist_tol) reached at t = $(integrator.t). Terminating solver")
        terminate!(integrator)
    end

    # Build callbacks
    step_cb = FunctionCallingCallback(propagate_U_and_update_distance!; func_everystep=true)
    event_cb = DiscreteCallback(condition, affect!)
    cb = CallbackSet(step_cb, event_cb)

    # Solution 
    sol = solve(prob, method; callback=cb, reltol=solver.reltol, abstol=solver.abstol, adaptive=false, kwargs...)

    if full_results 
        return sol, tracker
    else 
        return tracker.min_dist
    end 
end 

#############
function find_best_initial_costate(tar::TargetContainer, algebra::Algebra, solver::SolverParams, stor::Storage, method=Midpoint(); m0=nothing, show_trace=true, show_every=50, iterations=1000, kwargs...)

    # check target before propagation
    # check_unitarity(tar.target, tar.tmp_for_target, solver.unitary_tol)
    # targ_dist = distance(tar.target, tar, algebra, stor)
    # @assert targ_dist < solver.dist_tol "Error in target overlap: $targ_dist"

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    objective = function(m) 
        m ./= norm(m)
        propagate(m, algebra, solver, stor, tar, method; save_everystep = false,
        save_start = false,
        save_end = false,
        dense = false, 
        kwargs...)
    end

    result = Optim.optimize(objective, m0, NelderMead(), Optim.Options(
        callback = state -> state.f_lowest < solver.dist_tol,
        show_trace  = show_trace, 
        g_abstol = solver.grad_tol,
        show_every = show_every,
        iterations=iterations
    ))
    m_best = result.minimizer

    return m_best
end
