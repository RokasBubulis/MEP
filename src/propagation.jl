using DifferentialEquations, OrdinaryDiffEqLowOrderRK
using DiffEqCallbacks, FiniteDiff, ADTypes
using Printf

include("adm.jl")
include("distance.jl")

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

function propagate(m0::AbstractVector, algebra::Algebra, solver::SolverParams, tar::TargetContainer, method=Midpoint(); save = false, kwargs...)

    stor = Storage(algebra.n_particles, length(algebra.lie_basis))
    @assert haskey(kwargs, :dt) "Non-adaptive methods must be assigned a fixed time step"
    # Initial values
    m0_arr = zeros(Float64, length(algebra.lie_basis))
    m0_arr[2:end] = m0
    x0 = m0_arr

    # set up problem
    stor.U .= stor.U0
    # d0 = distance(stor.U, tar, algebra, stor)
    # tracker = OutputTracker(d0, 0.0)
    tracker = OutputTracker(1.0, solver.tmax)
    params = algebra, tar, solver, stor, tracker
    prob = ODEProblem(f!, x0, (0.0, solver.tmax), params)
    if save
        n = size(algebra.im_control, 1)
        n_steps = round(Int, solver.tmax / kwargs[:dt])
        # the solver may split the final step to land exactly on tmax, so allow
        # one spare slot beyond the t = 0 state and the tmax/dt steps
        Ulist = [Matrix{ComplexF64}(undef, n, n) for _ in 1:n_steps+2]
        αlist = zeros(Float64, n_steps+2)
    else
        Ulist = nothing 
        αlist = nothing
    end 
    # number of entries written to Ulist/αlist; the tail stays unset on early termination
    saved = Ref(0)
    
    # Step-wise computation to update minimum distance to target coset
    first_step = Ref(true)
    function propagate_U_and_update_distance!(u, t, integrator; save=false, ulist=Ulist, alphalist=αlist)
        tar, solver, stor, tracker = integrator.p[2], integrator.p[3], integrator.p[4], integrator.p[5]

        optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, u, algebra, stor)
        Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
        #check_anti_hermiticity(stor.H_opt)

        # The callback also fires at t = 0 (func_start), where no interval has been
        # traversed yet. Advancing there would leave stor.U one step ahead of u, so
        # the distance below — and every entry of ulist — would be mislabelled by dt.
        if first_step[]
            first_step[] = false
            if save
                saved[] += 1
                ulist[saved[]] .= stor.U
                alphalist[saved[]] = stor.alpha
            end
            return nothing
        end

        copyto!(stor.dU, stor.H_opt)
        lmul!(integrator.dt, stor.dU)
        exponential!(stor.dU, stor.exp_method, stor.exp_cache)
        mul!(stor.U_buffer, stor.dU, stor.U)
        stor.U .= stor.U_buffer
        check_unitarity(stor.U, stor.U_adj_buffer, stor.U_unitary_buffer_check, solver.unitary_tol)

        # stor.U now corresponds to time t, matching u
        if save
            saved[] += 1
            ulist[saved[]] .= stor.U
            alphalist[saved[]] = stor.alpha
        end 

        dist = distance(stor.U, tar, algebra, stor)
        if dist < 0.0
            @warn("Negative dist to target coset obtained: $dist, setting to positive")
            dist = abs(dist)
        end 
        if dist < tracker.min_dist && solver.tstar_min <= t <= solver.tstar_max
            tracker.min_dist = dist 
            tracker.tstar = t 
        end 
    end

    # Functions to check if solution can be terminated
    function condition(u, t, integrator)
        return integrator.p[5].min_dist < integrator.p[3].dist_tol
    end

    function affect!(integrator)
        #println("Distance $(integrator.p[5].min_dist) below tolerance $(integrator.p[3].dist_tol) reached at t = $(integrator.t). Terminating solver")
        terminate!(integrator)
    end

    # Build callbacks
    step_cb = FunctionCallingCallback(func_everystep=true) do u, t, integrator
        propagate_U_and_update_distance!(u, t, integrator; save=save, ulist=Ulist, alphalist=αlist)
    end
    event_cb = DiscreteCallback(condition, affect!)
    cb = CallbackSet(step_cb, event_cb)

    # Solution 
    sol = solve(prob, method; callback=cb, reltol=solver.reltol, abstol=solver.abstol, adaptive=false, kwargs...)

    if save 
        # early termination leaves the tail unpopulated
        return tracker.min_dist, tracker.tstar, Ulist[1:saved[]], αlist[1:saved[]]
    else 
        return tracker.min_dist
    end 
end 

function find_best_initial_costate(tar::TargetContainer, algebra::Algebra, solver::SolverParams, method=Midpoint(); m0=nothing, show_every=50, iterations=1000, kwargs...)

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    iter = Ref(0)
    objective = function(m) 
        iter[] += 1
        cost, tstar, _, _ = propagate(m, algebra, solver, tar, method; save = true,
        save_everystep = false,
        save_start = false,
        save_end = false,
        dense = false, 
        kwargs...)
        @printf("iter = %4d  ->  cost = %.8f  at t* = %.6f\n", iter[], cost, tstar)
        return cost
    end

    result = Optim.optimize(objective, m0, NelderMead(), Optim.Options(
        callback = state -> state.f_lowest < solver.dist_tol,
        g_abstol = solver.grad_tol,
        show_every = show_every,
        iterations=iterations
    ))
    m_best = result.minimizer

    dmin, tstar, _, _ = propagate(m_best, algebra, solver, tar, method; save = true,
    save_everystep = false,
    save_start = false,
    save_end = false,
    dense = false, 
    kwargs...)
    @printf("Final: cost = %.8f  at t* = %.6f  after %d iterations\n",
        dmin, tstar, iter[])

    return m_best
end
