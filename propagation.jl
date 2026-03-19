# using DifferentialEquations

include("adjoint_drift_maximisation.jl")
include("checks.jl")

function build_M!(m, params)
    M = params.storage_params.M_tmp
    M .= 0
    for (i, m_coeff) in enumerate(m)
        M.+= m_coeff * params.derived_args.p_basis[i]
    end
    #M /= norm(M)
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

function propagate(m, params)
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
    
    dmin = min_dist_to_target_coset(stor.U_tmp, params)
    tstar = prop.tmin

    # check target before propagation
    check_unitarity(params.system_params.target, 0; note="Target")
    target_dist = min_dist_to_target_coset(params.system_params.target, params)
    @assert target_dist < prop.coset_tol "dist(target) = $target_dist"

    for (i, ti) in enumerate(ts)
        Us[i] = copy(stor.U_tmp)
        Ms[i] = copy(stor.M_tmp)
        # check initial and following actions
        check_unitarity(stor.U_tmp, i)
        dists[i] = min_dist_to_target_coset(stor.U_tmp, params)

        α_opt = optimal_adjoint_drift_newton!(stor.M_tmp, params)
        αs[i] = α_opt
        Hs[i] = copy(stor.adjoint_drift_tmp)
        stor.U_tmp .= exp(stor.adjoint_drift_tmp * prop.dt) * stor.U_tmp
        #mul!(stor.U_tmp, exp(stor.adjoint_drift_tmp .* dt), stor.U_tmp)
        br_for_dM!(params)
        stor.M_tmp .+= stor.dM .* prop.dt

        dist = dists[i] 

        if dist < prop.coset_tol
            # return ts[1:i], Us[1:i], Ms[1:i], dists[1:i]
            return dist, ti
        
        elseif prop.coset_tol < dist < dmin 
            dmin = dist 
            tstar = ti 
        end 
    end 

    return dmin, tstar
    # return ts, Us, Ms, dists, Hs, αs
end


