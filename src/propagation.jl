using DifferentialEquations, OrdinaryDiffEqLowOrderRK
using DiffEqCallbacks, FiniteDiff, ADTypes
using CMAEvolutionStrategy
using BlackBoxOptim

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
        Ulist = [Matrix{ComplexF64}(undef, n, n) for _ in 1:n_steps+2]  # add extra step due to floating point precision 
        αlist = zeros(Float64, n_steps+2)
        Ulist[1] = copy(stor.U)
        αlist[1] = 0.0
    else
        Ulist = nothing 
        αlist = nothing
    end 
    
    # Step-wise computation to update minimum distance to target coset
    function propagate_U_and_update_distance!(u, t, integrator; save=false, ulist=Ulist, alphalist=αlist)
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
        if save
            ulist[integrator.iter+1] .= stor.U
            alphalist[integrator.iter+1] = stor.alpha
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
        return tracker.min_dist, tracker.tstar, Ulist[1:end-1], αlist[1:end-1]
    else 
        return tracker.min_dist
    end 
end 

# #############
# function find_best_initial_costate_bbo(tar::TargetContainer, algebra::Algebra, solver::SolverParams, method=Midpoint(); max_evals=10000, verbose=false, logging=false, kwargs...)

#     m_log = Vector{Vector{Float64}}()
#     if logging 
#         objective = function(m) 
#             push!(m_log, copy(m))
#             propagate(m, algebra, solver, tar, method; save_everystep = false,
#             save_start = false,
#             save_end = false,
#             dense = false, 
#             kwargs...)
#         end 
#     else 
#         objective = function(m) 
#             # m ./= norm(m)
#             propagate(m, algebra, solver, tar, method; save_everystep = false,
#             save_start = false,
#             save_end = false,
#             dense = false, 
#             kwargs...)
#         end
#     end 
#     if verbose
#         trace_flag = :compact 
#     else 
#         trace_flag = :silent 
#     end 
#     result = bboptimize(objective;
#         #Method = :adaptive_de_rand_1_bin_radiuslimited, 
#         SearchRange = (-1.0, 1.0),
#         NumDimensions = length(algebra.p_basis),
#         MaxFuncEvals = max_evals,
#         TargetFitness = 0.0,
#         FitnessTolerance = 1e-5,
#         TraceMode = trace_flag,
#         TraceInterval = 10.0)

#     return best_candidate(result), best_fitness(result), m_log
# end

function find_best_initial_costate_bbo(tar::TargetContainer, algebra::Algebra, solver::SolverParams, method=Midpoint(); max_evals=10000, verbose=false, logging=false, kwargs...)

    m_log = Vector{Vector{Float64}}()
    best_fitness_so_far = Ref(Inf)

    if logging
        objective = function(m)
            val = propagate(m, algebra, solver, tar, method; save_everystep = false,
                save_start = false,
                save_end = false,
                dense = false,
                kwargs...)
            if val <= best_fitness_so_far[]
                best_fitness_so_far[] = val
                push!(m_log, copy(m))
            end
            return val
        end
    else
        objective = function(m)
            propagate(m, algebra, solver, tar, method; save_everystep = false,
                save_start = false,
                save_end = false,
                dense = false,
                kwargs...)
        end
    end

    if verbose
        trace_flag = :compact
    else
        trace_flag = :silent
    end

    result = bboptimize(objective;
        PopulationSize = 100,
        SearchRange = (-1.0, 1.0),
        NumDimensions = length(algebra.p_basis),
        MaxFuncEvals = max_evals,
        TargetFitness = 0.0,
        FitnessTolerance = 1e-5,
        TraceMode = trace_flag,
        TraceInterval = 30.0)

    return best_candidate(result), best_fitness(result), m_log
end

function find_best_initial_costate(tar::TargetContainer, algebra::Algebra, solver::SolverParams, method=Midpoint(); m0=nothing, show_trace=true, show_every=50, iterations=1000, kwargs...)

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    objective = function(m) 
        #m ./= norm(m)
        propagate(m, algebra, solver, tar, method; save_everystep = false,
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

function find_best_initial_costate_with_logging(tar::TargetContainer, algebra::Algebra, solver::SolverParams, stor::Storage, method=Midpoint(); m0=nothing, show_trace=true, show_every=50, iterations=1000, kwargs...)

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    m_log = Vector{Vector{Float64}}()
    objective = function(m) 
        m ./= norm(m)
        push!(m_log, copy(m))
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

    return m_best, m_log
end


function find_best_initial_costate_cma(tar::TargetContainer, algebra::Algebra, solver::SolverParams, method=Midpoint(); m0=nothing, show_trace=true, show_every=50, iterations=1000, sigma=0.1, kwargs...)

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    objective = function(m) 
        #m ./= norm(m)
        propagate(m, algebra, solver, tar, method; save_everystep = false,
        save_start = false,
        save_end = false,
        dense = false, 
        kwargs...)
    end

    result = minimize(
        objective,
        m0,
        sigma;                  
        maxiter = iterations,
        ftarget = solver.dist_tol,
        verbosity = show_trace ? 1 : 0
    )

    return xbest(result), fbest(result)
end


function find_best_initial_costate_cma_with_logging(tar::TargetContainer, algebra::Algebra, solver::SolverParams, stor::Storage, method=Midpoint(); m0=nothing, show_trace=true, sigma=0.1, iterations=1000, kwargs...)

    if isnothing(m0)
        n = length(algebra.p_basis)
        m0 = zeros(n)
        m0[1] = 1.0
    end 

    m_log = Vector{Vector{Float64}}()
    objective = function(m) 
        m ./= norm(m)
        push!(m_log, copy(m))
        propagate(m, algebra, solver, stor, tar, method; save_everystep = false,
        save_start = false,
        save_end = false,
        dense = false, 
        kwargs...)
    end

    result = minimize(
        objective,
        m0,
        sigma;                  
        maxiter = iterations,
        ftarget = solver.dist_tol,
        verbosity = show_trace ? 1 : 0
    )

    m_best = xbest(result)

    return m_best, m_log
end