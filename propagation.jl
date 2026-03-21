
include("set_params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")

function build_M0_first_order!(M0, m, params)
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
    
    res = optimize(dist, -pi, pi)
    β_opt = Optim.minimizer(res)

    return dist(β_opt)
end

# function br_for_dM!(params)
#     stor = params.storage_params
#     mul!(stor.tmp1, stor.adjoint_drift_tmp, stor.M_tmp)
#     mul!(stor.tmp2, stor.M_tmp, stor.adjoint_drift_tmp)
#     stor.dM .= stor.tmp1 - stor.tmp2
#     return nothing 
# end

function objective(m, params)
    # propagation accurate to first order in dt, use M_tmp0 only
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(prop.U0)
    build_M0_first_order!(stor.M_tmp0, m, params)
    ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))
    
    dmin = min_dist_to_target_coset(stor.U_tmp, params)
    tstar = prop.tmin

    for (i, ti) in enumerate(ts)
        # check initial and following actions
        check_unitarity(stor.U_tmp, i)
        dist = min_dist_to_target_coset(stor.U_tmp, params)

        # compute H_opt
        optimal_adjoint_drift_newton!(stor.M_tmp0, params)

        # U(t+dt) = exp(H_opt * dt) * U(t)
        mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
        stor.U_tmp .= stor.dU

        # dM/dt = [H_opt, M(t)]
        mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp0)
        mul!(stor.tmp_mat, stor.M_tmp0, stor.adjoint_drift_tmp)
        stor.dM .-= stor.tmp_mat

        # M(t+dt) = [H_opt, M(t)]*dt + M(t-dt)
        stor.M_tmp0 .+= stor.dM .* prop.dt

        if dist < prop.coset_tol
            return dist, ti
        
        elseif prop.coset_tol < dist < dmin 
            dmin = dist 
            tstar = ti 
        end 
    end 

    return dmin
end

function propagate_and_store_results(m, params)
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(prop.U0)
    build_M0_first_order!(stor.M_tmp0, m, params)
    ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))

    n = length(ts)
    Us = Vector{typeof(stor.U_tmp)}(undef, n)
    Ms = Vector{typeof(stor.M_tmp0)}(undef, n)
    Hs = Vector{typeof(stor.U_tmp)}(undef, n)

    for i in eachindex(ts)
        # store
        Us[i] = copy(stor.U_tmp)
        Ms[i] = copy(stor.M_tmp0)
        Hs[i] = copy(stor.adjoint_drift_tmp)

        # check initial and following actions
        check_unitarity(stor.U_tmp, i)

        # compute H_opt
        optimal_adjoint_drift_newton!(stor.M_tmp0, params)

        # U(t+dt) = exp(H_opt * dt) * U(t)
        mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
        stor.U_tmp .= stor.dU

        # dM/dt = [H_opt, M(t)]
        mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp0)
        mul!(stor.tmp_mat, stor.M_tmp0, stor.adjoint_drift_tmp)
        stor.dM .-= stor.tmp_mat

        # M(t+dt) = [H_opt, M(t)]*dt + M(t-dt)
        stor.M_tmp0 .+= stor.dM .* prop.dt
    end 

    return ts, Us, Ms, Hs
end


# function objective_2nd_order(m, params)
#     prop = params.propagation_params
#     stor = params.storage_params

#     stor.U_tmp = copy(prop.U0)
#     build_M!(m, params)
#     ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))
    
#     dmin = min_dist_to_target_coset(stor.U_tmp, params)
#     tstar = prop.tmin

#     # check target before propagation TODO move outside the function
#     check_unitarity(params.system_params.target, 0; note="Target")
#     target_dist = min_dist_to_target_coset(params.system_params.target, params)
#     @assert target_dist < prop.coset_tol "dist(target) = $target_dist"

#     for (i, ti) in enumerate(ts)
#         # check initial and following actions
#         check_unitarity(stor.U_tmp, i)
#         dist = min_dist_to_target_coset(stor.U_tmp, params)

#         # obtain H_opt
#         optimal_adjoint_drift_newton!(stor.M_tmp1, params)
#         # U(t+dt) = exp(H_opt*dt) * U(t)
#         mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
#         stor.U_tmp .= stor.dU

#         mul!(stor.dM, stor.adjoint_drift_tmp, stor.M_tmp1)  # H_opt * M
#         mul!(stor.tmp_mat, stor.M_tmp1, stor.adjoint_drift_tmp)  # M * H_opt
#         stor.dM .-= stor.tmp_mat  # dM/dt = H_opt * M(t) - M(t) * H_opt
#         # M(t+dt) = 2*dt*dM/dt + M(t-dt)
#         stor.M_tmp2 .= 2 * prop.dt .* stor.dM .+ stor.tmp0

#         if dist < prop.coset_tol
#             return dist, ti
        
#         elseif prop.coset_tol < dist < dmin 
#             dmin = dist 
#             tstar = ti 
#         end 
#         stor.M_tmp0 .= copy(stor.M_tmp1)
#         stor.M_tmp1 .= copy(stor.M_tmp2)
#     end 

#     return dmin
# end

