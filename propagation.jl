
include("set_params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")

function build_M0!(M0, m, params)
    M0 .= 0
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * params.derived_args.p_basis[i]
    end
    M0 /= norm(M0)
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

function propagator_1st_order_step!(params)

    stor = params.storage_params
    prop = params.propagation_params

    # compute H_opt
    optimal_adjoint_drift_newton!(stor.M_tmp0, params)

    # U(t+dt) = exp(H_opt * dt) * U(t)
    mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
    stor.U_tmp .= stor.dU

    # dM/dt = [H_opt, M(t)]
    mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp0)
    mul!(stor.tmp_mat, stor.M_tmp0, stor.adjoint_drift_tmp)
    stor.dM .-= stor.tmp_mat

    # M(t+dt) = [H_opt, M(t)]*dt + M(t)
    stor.M_tmp0 .+= stor.dM .* prop.dt

    return nothing
end 

function propagate(m, params)
    # propagation accurate to first order in dt, use M_tmp0 only
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(prop.U0)
    build_M0!(stor.M_tmp0, m, params)
    ts = collect(range(0.0, prop.tmax; step=prop.dt))
    
    dmin = min_dist_to_target_coset(stor.U_tmp, params)
    tstar = prop.tmin

    for (i, ti) in enumerate(ts)

        dist = min_dist_to_target_coset(stor.U_tmp, params)

        if dist < prop.coset_tol
            return dist, ti
        
        elseif prop.coset_tol < dist < dmin 
            dmin = dist 
            tstar = ti 
        end 

        # check initial and following actions
        check_unitarity(stor.U_tmp, i)

        propagator_1st_order_step!(params)

    end 

    return dmin
end

function propagate_and_store_results_1st_order(m, params)
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(prop.U0)
    build_M0!(stor.M_tmp0, m, params)
    ts = collect(range(0.0, prop.tmax; step=prop.dt))

    n = length(ts)
    Us = Vector{typeof(stor.U_tmp)}(undef, n)
    Ms = Vector{typeof(stor.M_tmp0)}(undef, n)
    dists = Vector{Float64}(undef, n)

    for i in eachindex(ts)

        # compute H_opt
        optimal_adjoint_drift_newton!(stor.M_tmp0, params)

        # store
        Us[i] = copy(stor.U_tmp)
        Ms[i] = copy(stor.M_tmp0)
        dists[i] = min_dist_to_target_coset(stor.U_tmp, params)

        # check initial and following actions
        check_unitarity(stor.U_tmp, i)

        propagator_1st_order_step!(params)

    end 

    return ts, Us, Ms, dists
end

function set_initial_state_2nd_order!(m, params)
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    prop = params.propagation_params
    stor = params.storage_params

    build_M0!(stor.M_tmp0, m, params)  # M(0)
    optimal_adjoint_drift_newton!(stor.M_tmp0, params)  # H_opt(0)

    # M(dt) = [H_opt(0), M(0)] * dt
    mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp0)
    mul!(stor.tmp_mat, stor.M_tmp0, stor.adjoint_drift_tmp)
    stor.dM .-= stor.tmp_mat
    stor.M_tmp1 .= stor.dM * prop.dt .+ stor.M_tmp0

    mul!(stor.U_tmp, exp(stor.adjoint_drift_tmp * prop.dt), copy(prop.U0))  # U(dt)

    return nothing
end

function propagator_2nd_order_step!(params)
    stor = params.storage_params
    prop = params.propagation_params

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift_newton!(stor.M_tmp1, params)

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

function propagate_2nd_order(m, params; store=false)
    prop = params.propagation_params
    stor = params.storage_params
    # if store
    #     prop.dt = prop.dt_base / 10
    # else
    #     prop.dt = prop.dt_base
    # end
    set_initial_state_2nd_order!(m, params)
    ts = collect(range(0.0, prop.tmax; step=prop.dt))
    n = length(ts)

    d1 = min_dist_to_target_coset(prop.U0, params)
    d2 = min_dist_to_target_coset(stor.U_tmp, params)
    dmin = min(d1, d2)

    if store
        Us = Vector{typeof(stor.U_tmp)}(undef, n)
        Ms = Vector{typeof(stor.M_tmp0)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(prop.U0)
        Us[2] = copy(stor.U_tmp)
        Ms[1] = copy(stor.M_tmp0)
        Ms[2] = copy(stor.M_tmp1)
        dists[1] = d1
        dists[2] = d2
    end

    for i in eachindex(ts)[3:end]
        dist = min_dist_to_target_coset(stor.U_tmp, params)

        if !store
            if dist < prop.coset_tol
                return dist
            elseif dist < dmin
                dmin = dist
            end

            # prop.dt = if dist < 0.01
            #     prop.dt_base / 100
            # elseif dist < 0.1
            #     prop.dt_base / 10
            # else
            #     prop.dt_base
            # end
        end

        check_unitarity(stor.U_tmp, i)
        check_belongs_to_p_basis(stor.M_tmp0, params, i)
        propagator_2nd_order_step!(params)

        if store
            Us[i] = copy(stor.U_tmp)
            Ms[i] = copy(stor.M_tmp1)
            dists[i] = min_dist_to_target_coset(stor.U_tmp, params)
        end
    end

    return store ? (ts, Us, Ms, dists) : dmin
end