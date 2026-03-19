
include("set_params.jl")
include("adjoint_drift_maximisation.jl")
include("checks.jl")

function build_M!(m, params)
    M = params.storage_params.M_tmp
    M .= 0
    for (i, m_coeff) in enumerate(m)
        M.+= m_coeff * params.derived_args.p_basis[i]
    end
    M /= norm(M)
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

function br_for_dM!(params)
    stor = params.storage_params
    mul!(stor.tmp1, stor.adjoint_drift_tmp, stor.M_tmp)
    mul!(stor.tmp2, stor.M_tmp, stor.adjoint_drift_tmp)
    stor.dM .= stor.tmp1 - stor.tmp2
    return nothing 
end

function objective(m, params)
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(prop.U0)
    build_M!(m, params)
    ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))
    
    dmin = min_dist_to_target_coset(stor.U_tmp, params)
    tstar = prop.tmin

    # check target before propagation
    check_unitarity(params.system_params.target, 0; note="Target")
    target_dist = min_dist_to_target_coset(params.system_params.target, params)
    @assert target_dist < prop.coset_tol "dist(target) = $target_dist"

    for (i, ti) in enumerate(ts)
        # check initial and following actions
        check_unitarity(stor.U_tmp, i)
        dist = min_dist_to_target_coset(stor.U_tmp, params)

        optimal_adjoint_drift_newton!(stor.M_tmp, params)
        #stor.U_tmp .= exp(stor.adjoint_drift_tmp * prop.dt) * stor.U_tmp
        mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
        stor.U_tmp .= stor.dU
        br_for_dM!(params)
        stor.M_tmp .+= stor.dM .* prop.dt

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
    build_M!(m, params)
    ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))

    n = length(ts)
    Us = Vector{typeof(stor.U_tmp)}(undef, n)
    Ms = Vector{typeof(stor.M_tmp)}(undef, n)
    Hs = Vector{typeof(stor.U_tmp)}(undef, n)
    αs = Vector{Float64}(undef, n)
    dists = Vector{Float64}(undef, n)

    # check target before propagation
    check_unitarity(params.system_params.target, 0; note="Target")
    target_dist = min_dist_to_target_coset(params.system_params.target, params)
    @assert target_dist < prop.coset_tol "dist(target) = $target_dist"

    for i in eachindex(ts)

        dist = min_dist_to_target_coset(stor.U_tmp, params)

        # store
        Us[i] = copy(stor.U_tmp)
        Ms[i] = copy(stor.M_tmp)
        Hs[i] = copy(stor.adjoint_drift_tmp)
        dists[i] = dist

        # check initial and following actions
        check_unitarity(stor.U_tmp, i)
        
        # propagate
        optimal_adjoint_drift_newton!(stor.M_tmp, params)
        mul!(stor.dU, exp(stor.adjoint_drift_tmp .* prop.dt), stor.U_tmp)
        stor.U_tmp .= stor.dU
        br_for_dM!(params)
        stor.M_tmp .+= stor.dM .* prop.dt 
    end 

    return ts, Us, Ms, Hs, dists
end


