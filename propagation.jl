# using DifferentialEquations

include("adjoint_drift_maximisation.jl")

function build_M!(m, params)
    M = params.storage_params.M_tmp
    M .= 0 #[:]
    # fill!(M, zero(T))
    for (i, m_coeff) in enumerate(m)
        M.+= m_coeff * params.derived_args.p_basis[i] # M[:] fails with DimensionMismatch
    end
    M /= norm(M)
    return nothing
end 

# if diagonal control
function min_dist_to_target_coset(params)
    A = params.storage_params.U_tmp * adjoint(params.system_params.target)

    dist(β) = norm(A - spdiagm(0 => exp.(β.* params.derived_args.diag_im_control_vec)))
    
    res = optimize(dist, -pi, pi)
    β_opt = Optim.minimizer(res)

    return dist(β_opt)
end

function br_for_dM!(params)
    stor = params.storage_params
    mul!(stor.tmp1, stor.adjoint_drift_tmp, stor.M_tmp)
    mul!(stor.tmp2, stor.M_tmp, stor.adjoint_drift_tmp)
    stor.dM .= stor.tmp1 - stor.tmp2 # dM[:] fails with DimensionMismatch
    return nothing 
end

function differential!(params)
    stor = params.storage_params
    optimal_adjoint_drift_newton!(stor.M_tmp, params)
    mul!(stor.dU, stor.adjoint_drift_tmp, stor.U_tmp)
    br_for_dM!(params)
    return nothing 
end

function propagate(m, params)
    prop = params.propagation_params
    stor = params.storage_params

    stor.U_tmp = copy(params.propagation_params.U0)
    build_M!(m, params)
    ts = collect(range(prop.tmin, prop.tmax; step=prop.dt))
    n = length(ts)

    Us = Vector{typeof(stor.U_tmp)}(undef, n)
    dists = Vector{Float64}(undef, n)
    
    dmin = min_dist_to_target_coset(params)
    tstar = prop.tmin
    M = params.storage_params.M_tmp

    for (i, ti) in enumerate(ts)
        Us[i] = copy(stor.U_tmp)
        dists[i] = min_dist_to_target_coset(params)
        
        differential!(params)
        stor.U_tmp .+= stor.dU * prop.dt
        stor.M_tmp .+= stor.dM * prop.dt 

        dist = dists[i] 

        if dist < prop.coset_tol
            return ts[1:i], Us[1:i], dists[1:i]
        
        elseif prop.coset_tol < dist < dmin 
            dmin = dist 
            tstar = ti 
        end 
    end 

    return ts, Us, dists 
end


