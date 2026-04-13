include("params.jl")

function build_M0!(M0::Matrix{T}, m::Vector{Float64}, params::Params)
    M0 .= 0
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * params.algebra.p_basis[i]
    end
    #M0 /= norm(M0)
    return nothing
end 

# # if diagonal control
# function min_dist_to_target_coset(U, params)
#     A = U * adjoint(params.system_params.target)

#     dist(β) = norm(A - spdiagm(0 => exp.(β.* params.derived_args.diag_im_control_vec)))
    
#     res = optimize(dist, -pi, pi)  # decreasing tolerance does not help
#     β_opt = Optim.minimizer(res)

#     return dist(β_opt)
# end

function distance_objective(U::Union{Matrix{T}, SparseMatrixCSC{T,Int}}, 
    params::Params, stor::StorageParams)
    # 1 - overlap between U and target
    mul!(stor.tmp, U, params.physics.adjoint_target)
    return 1.0 - abs(tr(stor.tmp))/size(U,1)
end

### Propagation ### 

function set_initial_state_2nd_order!(m::Vector{Float64}, params::Params, stor::StorageParams)
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    build_M0!(stor.M0, m, params)  # M(0)
    optimal_adjoint_drift_newton!(stor.adjoint_drift, stor.M0, params)  # H_opt(0)

    # M(dt) = [H_opt(0), M(0)] * dt
    mul!(stor.dM, stor.adjoint_drift, stor.M0)
    mul!(stor.tmp, stor.M0, stor.adjoint_drift)
    stor.dM .-= stor.tmp
    stor.M1 .= stor.dM * params.solver.dt .+ stor.M0

    mul!(stor.U, exp(stor.adjoint_drift * params.solver.dt), copy(params.U0))  # U(dt)

    return nothing
end

function propagator_2nd_order_step!(params::Params, stor::StorageParams)

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift_newton!(stor.adjoint_drift, stor.M1, params)

    # U(t+dt) = exp(H_opt * dt) * U(t)
    mul!(stor.dU, exp(stor.adjoint_drift .* params.solver.dt), stor.U)
    stor.U[:] .= stor.dU[:]

    # dM/dt = [H_opt, M(t)]
    mul!(stor.dM, stor.adjoint_drift, stor.M1)
    mul!(stor.tmp, stor.M1, stor.adjoint_drift)
    stor.dM[:] .-= stor.tmp[:]

    # M(t+dt) = 2*dt*[H_opt, M(t)] + M(t-dt)
    stor.M2[:] .= (2*params.solver.dt) .* stor.dM[:] .+ stor.M0[:]

    # M(t-dt) -> M(t)
    # M(t) -> M(t+dt)
    stor.M0[:] .= stor.M1[:]
    stor.M1[:] .= stor.M2[:]

    return nothing
end 

function propagate_2nd_order(m::Vector{Float64}, params::Params, stor::StorageParams; save=false) 
    # 574 μs without save and without checks, 630 μs with checks

    set_initial_state_2nd_order!(m, params, stor)
    ts = collect(range(0.0, params.solver.tmax; step=params.solver.dt))
    n = length(ts)

    d1 = distance_objective(params.U0, params, stor)
    d2 = distance_objective(stor.U, params, stor)
    dmin = min(d1, d2)

    if save
        Us = Vector{typeof(stor.U)}(undef, n)
        Ms = Vector{typeof(stor.M0)}(undef, n)
        Hs = Vector{typeof(stor.adjoint_drift)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(params.U0)
        Us[2] = copy(stor.U)
        Ms[1] = copy(stor.M0)
        Ms[2] = copy(stor.M1)
        Hs[1] = copy(stor.adjoint_drift)
        optimal_adjoint_drift_newton!(stor.tmp, stor.M1, params)
        Hs[2] = copy(stor.tmp)
        dists[1] = d1
        dists[2] = d2
    end

    for i in eachindex(ts)[3:end]
        dist = distance_objective(stor.U, params, stor)

        if !save
            if dist < params.solver.tol
                return dist
            elseif dist < dmin
                dmin = dist
            end
        end

        check_unitarity(stor.U, i)
        check_costate(stor.M0, params, i)
        propagator_2nd_order_step!(params, stor)

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M1)
            Hs[i] = copy(stor.adjoint_drift)
            dists[i] = dist
        end
    end

    return save ? (ts, Us, Ms, Hs, dists) : dmin
end

### Shooting approach to find the best initial costate M0 ###

# represent the search space as n-1 angles on the hyper-sphere ensuring |M(t)| = 1
function angles_to_directions(angles::AbstractVector{Float64})
    n = length(angles) + 1
    m = zeros(n)
    m[1] = cos(angles[1])
    prefix = sin(angles[1])
    for i in 2:n-1
        m[i] = prefix * cos(angles[i])
        prefix *= sin(angles[i])
    end
    m[n] = prefix
    return m
end

function find_best_initial_costate(params::Params, stor::StorageParams)

    # check target before propagation
    check_unitarity(params.physics.target, 0; note="Target")
    targ_dist = distance_objective(params.physics.target, params, stor)
    @assert targ_dist < params.solver.tol "Error in target overlap: $targ_dist"

    n = length(params.algebra.p_basis) - 1
    initial_angles = zeros(n)

    #thread_stors = [deepcopy(stor) for _ in 1:Threads.nthreads()]
    thread_stors = [deepcopy(stor) for _ in 1:Threads.maxthreadid()]

    objective = function(angles)
        s = thread_stors[Threads.threadid()]
        propagate_2nd_order(angles_to_directions(angles), params, s)
    end

    #objective(angles) = propagate_2nd_order(angles_to_directions(angles), params, stor)

    # use CMA Evolution strategy for optimisation
    result = minimize(
        objective,
        initial_angles, 0.5;
        maxiter=500, verbosity=1,
        multi_threading=true,
        popsize=24,
        lower = zeros(n),
        upper = [i == 1 ? 2π : π for i in 1:n],
        ftarget=params.solver.tol
        )
    angles_best = xbest(result)
    m_best = angles_to_directions(angles_best)

    return m_best
end