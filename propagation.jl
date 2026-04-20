include("params.jl")

using ForwardDiff

function check_duals(x, name)
    if eltype(x) <: ForwardDiff.Dual || eltype(x) <: Complex{<:ForwardDiff.Dual}
        partials = ForwardDiff.partials.(real.(x))
        if all(iszero, partials)
            @warn "Dual parts are zero at $name"
        end
    end
end

function build_M0!(M0::Matrix{TC}, m::AbstractVector{TR}, params::Params) where {TR, TC}
    M0 .= 0
    check_duals(m, "m")
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * params.algebra.p_basis[i]
    end
    check_duals(M0, "M0")
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

# applicable only if diagonal control
function distance_objective_optimiser(U::Union{Matrix{T}, SparseMatrixCSC{T,Int}}, 
    params::Params, stor::StorageParams) where T

    mul!(stor.tmp, U, params.physics.adjoint_target)
    tmp_diag = diag(stor.tmp)

    dist(β) = 1 - 1/size(U,1) * abs(dot(exp.(β.*params.physics.im_control_vec), tmp_diag))
    # res = optimize(dist, -pi, pi)
    res = optimize(dist, -π, π, Brent(), 
    abs_tol = 1e-12,   # tighter argument tolerance
    rel_tol = 1e-12
)
    β_opt = Optim.minimizer(res)

    return dist(β_opt)
end

function distance_objective_analytic(β::TBeta, U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    params::Params, stor::StorageParams) where {TBeta, TCostate, TSystem}

    mul!(stor.tmp, U, params.physics.adjoint_target)
    A_diag = diag(stor.tmp) # A_jj 
    dim = size(stor.tmp, 1)
    expLBeta = exp.(β.*params.physics.im_control_vec)
    d = dot(expLBeta, A_diag)
    res =  1 - 1/dim * sqrt(real(d)^2 + imag(d)^2)
    return res
end

function distance_objective_analytic_derivatives(β::TBeta, U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    params::Params, stor::StorageParams) where {TBeta, TCostate, TSystem}

    first_der, second_der = zero(Float64), zero(Float64)
    dim = size(stor.tmp, 1)
    mul!(stor.tmp, U, params.physics.adjoint_target)
    A_diag = diag(stor.tmp) # A_jj 
    LexpBetaL = params.physics.im_control_vec .* exp.(β.*params.physics.im_control_vec)
    first_der = -1/dim * real(dot(LexpBetaL, A_diag))
    L2expBetaL = params.physics.im_control_vec .* LexpBetaL
    second_der = -1/dim * real(dot(L2expBetaL, A_diag))

    return first_der, second_der
end 

function minimum_distance_objective_analytic(U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    params::Params, stor::StorageParams) where {TCostate, TSystem}

    β = zero(Float64) # forward diff does not need to flow through beta
    for _ in 1:params.solver.Newton_steps
        first_der, second_der = distance_objective_analytic_derivatives(β, U, params, stor)
        dβ = first_der / second_der
        β -= dβ
        abs(dβ) < 1e-10 && break
    end 
    min_dist = distance_objective_analytic(β, U, params, stor)
    return min_dist
end     

distance(U::Matrix{<:Complex{<:ForwardDiff.Dual}}, params::Params, stor::StorageParams
) = minimum_distance_objective_analytic(U, params, stor)

distance(U::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64, Int}}, params::Params, stor::StorageParams
) = distance_objective_optimiser(U, params, stor)

### Propagation ### 

optimal_adjoint_drift!(tmp::Matrix{ComplexF64}, costate::Matrix{ComplexF64}, params::Params
) = optimal_adjoint_drift_optimiser!(tmp, costate, params)

optimal_adjoint_drift!(tmp::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
costate::Matrix{<:Complex{<:ForwardDiff.Dual}}, params::Params
) = optimal_adjoint_drift_analytic!(tmp, costate, params)

function exponent_analytic!(tmp::Matrix{T}, X::Matrix{T}; depth=20) where T

    copyto!(tmp, I)
    term = similar(tmp)
    copyto!(term, I)
    tmp_term = similar(tmp)
    for n in 1:depth
        mul!(tmp_term, term, X)
        term .= tmp_term / n
        tmp .+= term
    end
    return tmp
end

function exponent_builtin!(tmp::Matrix{T}, X::Matrix{T}) where T 
    tmp .= exp(X)
    return tmp
end 

exponent!(tmp::Matrix{ComplexF64}, X::Matrix{ComplexF64}) = exponent_builtin!(tmp, X)

exponent!(tmp::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
X::Matrix{<:Complex{<:ForwardDiff.Dual}}) = exponent_analytic!(tmp, X)

function set_initial_state_2nd_order!(m::AbstractVector{TR}, params::Params, stor::StorageParams) where TR
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    build_M0!(stor.M0, m, params)  # M(0)
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M0, params)  # H_opt(0)
    exponent!(stor.tmp, stor.adjoint_drift * params.solver.dt)
    mul!(stor.U, stor.tmp, copy(params.U0))  # U(dt)

    # M(dt) = [H_opt(0), M(0)] * dt
    mul!(stor.dM, stor.adjoint_drift, stor.M0)
    mul!(stor.tmp, stor.M0, stor.adjoint_drift)
    stor.dM .-= stor.tmp
    stor.M1 .= stor.dM * params.solver.dt .+ stor.M0

    return nothing
end

function propagator_2nd_order_step!(params::Params, stor::StorageParams)

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M1, params)

    # U(t+dt) = exp(H_opt * dt) * U(t)
    exponent!(stor.tmp, stor.adjoint_drift * params.solver.dt)
    mul!(stor.dU, stor.tmp, stor.U)
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

function propagate_2nd_order(m::AbstractVector{TR}, params::Params, stor::StorageParams; save=false) where TR
    # 574 μs without save and without checks, 630 μs with checks

    set_initial_state_2nd_order!(m, params, stor)
    ts = collect(range(0.0, params.solver.tmax; step=params.solver.dt))
    n = length(ts)

    d1 = distance(params.U0, params, stor)
    d2 = distance(stor.U, params, stor)
    dmin = min(d1, d2)
    check_duals(stor.U, "U(dt)")

    if save
        Us = Vector{typeof(stor.U)}(undef, n)
        Ms = Vector{typeof(stor.M0)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(params.U0)
        Us[2] = copy(stor.U)
        Ms[1] = copy(stor.M0)
        Ms[2] = copy(stor.M1)
        dists[1] = d1
        dists[2] = d2
    end

    for i in eachindex(ts)[3:end]

        check_unitarity(stor.U, stor.tmp, timestep=i)
        check_costate(stor.M0, params, i)
        propagator_2nd_order_step!(params, stor)
        dist = distance(stor.U, params, stor)

        check_duals(stor.M0, "M(t)")
        check_duals(stor.U, "U(t)")
        check_duals(dist, "dmin(t)")

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M1)
            dists[i] = dist
        else
            if dist < params.solver.tol
                return dist
            elseif dist < dmin
                dmin = dist
            end
        end
    end
    return save ? (ts, Us, Ms, dists) : dmin
end

### Shooting approach to find the best initial costate M0 ###

# represent the search space as n-1 angles on the hyper-sphere ensuring |M(t)| = 1
function angles_to_directions(angles::AbstractVector{TR}) where TR
    n = length(angles) + 1
    m = Vector{TR}(undef, n)
    m[1] = cos(angles[1])
    prefix = sin(angles[1])
    for i in 2:n-1
        m[i] = prefix * cos(angles[i])
        prefix *= sin(angles[i])
    end
    m[n] = prefix
    check_duals(m, "m(θ)")
    return m
end

function find_best_initial_costate_bbf(params::Params, stor::StorageParams)

    # check target before propagation
    check_unitarity(params.physics.target, stor.tmp, note="Target")
    targ_dist = distance(params.physics.target, params, stor)
    @assert targ_dist < params.solver.tol "Error in target overlap: $targ_dist"

    n = length(params.algebra.p_basis) - 1
    initial_angles = zeros(n)

    thread_stors = [deepcopy(stor) for _ in 1:Threads.maxthreadid()]

    objective = function(angles)
        s = thread_stors[Threads.threadid()]
        m = angles_to_directions(angles)
        list = [0, 1, 2, 2, 3, 4, 4]
        propagate_2nd_order(m, params, s) + params.solver.lambda *  dot(list, abs.(m).^2)
    end

    #objective(angles) = propagate_2nd_order(angles_to_directions(angles), params, stor)

    # use CMA Evolution strategy for optimisation
    result = minimize(
        objective,
        initial_angles, 1.0;
        maxiter=500, verbosity=1,
        multi_threading=true,
        popsize=24,
        lower = zeros(n),
        upper = [i == 1 ? 2π : π for i in 1:n],
        ftarget=params.solver.lambda != 0 ? params.solver.lambda * params.solver.tol : params.solver.tol
        )
    angles_best = xbest(result)
    m_best = angles_to_directions(angles_best)

    return m_best
end


function find_best_initial_costate_autograd(params::Params, stor::StorageParams)

    # check target before propagation
    check_unitarity(params.physics.target, stor.tmp, note="Target")
    targ_dist = distance(params.physics.target, params, stor)
    @assert targ_dist < params.solver.tol "Error in target overlap: $targ_dist"

    n = length(params.algebra.p_basis) - 1
    initial_angles = zeros(n)
    dim = size(stor.adjoint_drift, 1)

    stor_dual_ref = Ref{Any}(nothing)
    objective = function(angles)
        m = angles_to_directions(angles)
        if eltype(m) <: ForwardDiff.Dual
            T = Complex{eltype(m)}
            if typeof(stor_dual_ref[]) != StorageParams{T}
                stor_dual_ref[] = StorageParams{T}(dim)
            end
            storage = stor_dual_ref[]
        else
            storage = stor
        end
        propagate_2nd_order(m, params, storage)
    end

    od = OnceDifferentiable(objective, initial_angles; autodiff = :forward)
    angles_best = Optim.minimizer(
        optimize(
            od, initial_angles, BFGS(linesearch = BackTracking()), 
            Optim.Options(show_trace=true, f_abstol=params.solver.tol)
        )
    )

    # best_val = Inf
    # best_angles = initial_angles
    # for _ in 1:20
    #     x0 = rand(6) .* 2π
    #     od = OnceDifferentiable(objective, x0; autodiff=:forward)
    #     result = optimize(od, x0, BFGS(linesearch=BackTracking()),
    #                     Optim.Options(f_abstol=params.solver.tol))
    #     if Optim.minimum(result) < best_val
    #         best_val = Optim.minimum(result)
    #         best_angles = Optim.minimizer(result)
    #     end
    # end

    m_best = angles_to_directions(angles_best)

    return m_best
end