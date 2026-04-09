
function build_M0!(M0, m, params)
    M0 .= 0
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * params.derived_args.p_basis[i]
    end
    #M0 /= norm(M0)
    return nothing
end 

# if diagonal control
function min_dist_to_target_coset(U, params)
    A = U * adjoint(params.system_params.target)

    dist(β) = norm(A - spdiagm(0 => exp.(β.* params.derived_args.diag_im_control_vec)))
    
    res = optimize(dist, -pi, pi)  # decreasing tolerance does not help
    β_opt = Optim.minimizer(res)

    return dist(β_opt)
end

function distance_objective(U, params)
    # 1 - overlap between U and target
    return 1.0 - abs(tr(U*adjoint(params.system_params.target))/size(U,1))
end

### Propagation ### 

function set_initial_state_2nd_order!(m, params, stor)
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    prop = params.propagation_params

    build_M0!(stor.M_tmp0, m, params)  # M(0)
    optimal_adjoint_drift_newton!(stor.adjoint_drift_tmp, stor.M_tmp0, params)  # H_opt(0)

    # M(dt) = [H_opt(0), M(0)] * dt
    mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp0)
    mul!(stor.tmp_mat, stor.M_tmp0, stor.adjoint_drift_tmp)
    stor.dM .-= stor.tmp_mat
    stor.M_tmp1 .= stor.dM * prop.dt .+ stor.M_tmp0

    mul!(stor.U_tmp, exp(stor.adjoint_drift_tmp * prop.dt), copy(prop.U0))  # U(dt)

    return nothing
end

function propagator_2nd_order_step!(params, stor)
    prop = params.propagation_params

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift_newton!(stor.adjoint_drift_tmp, stor.M_tmp1, params)

    # U(t+dt) = exp(H_opt * dt) * U(t)
    mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
    stor.U_tmp[:] .= stor.dU[:]

    # dM/dt = [H_opt, M(t)]
    mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp1)
    mul!(stor.tmp_mat, stor.M_tmp1, stor.adjoint_drift_tmp)
    stor.dM[:] .-= stor.tmp_mat[:]

    # M(t+dt) = 2*dt*[H_opt, M(t)] + M(t-dt)
    stor.M_tmp2[:] .= (2*prop.dt) .* stor.dM[:] .+ stor.M_tmp0[:]

    # M(t-dt) -> M(t)
    # M(t) -> M(t+dt)
    stor.M_tmp0[:] .= stor.M_tmp1[:]
    stor.M_tmp1[:] .= stor.M_tmp2[:]

    return nothing
end 

function propagate_2nd_order(m, params, stor; save=false)
    prop = params.propagation_params

    set_initial_state_2nd_order!(m, params, stor)
    ts = collect(range(0.0, prop.tmax; step=prop.dt))
    n = length(ts)

    d1 = distance_objective(prop.U0, params)
    d2 = distance_objective(stor.U_tmp, params)
    dmin = min(d1, d2)

    if save
        Us = Vector{typeof(stor.U_tmp)}(undef, n)
        Ms = Vector{typeof(stor.M_tmp0)}(undef, n)
        Hs = Vector{typeof(stor.adjoint_drift_tmp)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(prop.U0)
        Us[2] = copy(stor.U_tmp)
        Ms[1] = copy(stor.M_tmp0)
        Ms[2] = copy(stor.M_tmp1)
        Hs[1] = copy(stor.adjoint_drift_tmp)
        optimal_adjoint_drift_newton!(stor.tmp_mat, stor.M_tmp1, params)
        Hs[2] = copy(stor.tmp_mat)
        dists[1] = d1
        dists[2] = d2
    end

    for i in eachindex(ts)[3:end]
        dist = distance_objective(stor.U_tmp, params)

        if !save
            if dist < prop.coset_tol
                return dist
            elseif dist < dmin
                dmin = dist
            end
        end

        check_unitarity(stor.U_tmp, i)
        check_costate(stor.M_tmp0, params, i)
        propagator_2nd_order_step!(params, stor)

        if save
            Us[i] = copy(stor.U_tmp)
            Ms[i] = copy(stor.M_tmp1)
            Hs[i] = copy(stor.adjoint_drift_tmp)
            dists[i] = dist
        end
    end

    return save ? (ts, Us, Ms, Hs, dists) : dmin
end

### Shooting approach to find the best initial costate M0 ###

# represent the search space as n-1 angles on the hyper-sphere ensuring |M(t)| = 1
function angles_to_directions(angles)
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

function find_best_initial_costate(params, stor)

    # check target before propagation
    check_unitarity(params.system_params.target, 0; note="Target")
    @assert distance_objective(params.system_params.target, params) < params.propagation_params.coset_tol "Error in target overlap"

    n = length(params.derived_args.p_basis) - 1
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
        ftarget=params.propagation_params.coset_tol
        )
    angles_best = xbest(result)
    m_best = angles_to_directions(angles_best)

    return m_best
end