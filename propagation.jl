include("adjoint_drift_maximisation.jl")
include("distance.jl")
include("adm_lie.jl")

function build_M0!(M0::Matrix{TC}, m::AbstractVector{TR}, algebra::Algebra) where {TR, TC}
    M0 .= 0
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * algebra.p_basis[i]
    end
    return nothing
end 

# function exponent_analytic!(tmp::Matrix{T}, X::Matrix{T}; depth=20) where T
function exponent_analytic!(tmp, X; depth=20)

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

function adjoint_action_series!(tmp, X, Y, stor)
    exponent_analytic!(stor.tmp1, X)
    exponent_analytic!(stor.tmp2, -X)
    mul!(stor.tmp3, Y, stor.tmp2)
    mul!(tmp, stor.tmp1, stor.tmp3)
    return nothing
end 

function exponent_builtin!(tmp::Matrix{T}, X::Matrix{T}) where T 
    tmp .= exp(X)
    return tmp
end 

exponent!(tmp::Matrix{ComplexF64}, X::Matrix{ComplexF64}) = exponent_builtin!(tmp, X)

exponent!(tmp::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
X::Matrix{<:Complex{<:ForwardDiff.Dual}}) = exponent_analytic!(tmp, X)

function set_initial_state_2nd_order!(m::AbstractVector{TR}, algebra::Algebra, system::System, 
    solver::SolverParams, stor::Storage) where TR
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    build_M0!(stor.M0, m, algebra)  # M(0)
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M0, algebra, system, solver, stor)  # H_opt(0)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.U, stor.tmp, stor.U0)  # U(dt)

    # M(dt) = [H_opt(0), M(0)] * dt + M0
    bracket_via_lie_coeffs!(stor.dM, stor.adjoint_drift, stor.M0, algebra, stor)
    stor.M1 .= stor.dM * solver.dt .+ stor.M0

    return nothing
end

function propagator_2nd_order_step!(algebra::Algebra, system::System, solver::SolverParams, stor::Storage)

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M1, algebra, system, solver, stor) 

    # U(t+dt) = exp(H_opt * dt) * U(t)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.dU, stor.tmp, stor.U)
    stor.U .= stor.dU

    # dM/dt = [H_opt, M(t)]
    bracket_via_lie_coeffs!(stor.dM, stor.adjoint_drift, stor.M1, algebra, stor)

    # M(t+dt) = 2*dt*[H_opt, M(t)] + M(t-dt)
    # stor.M2[:] .= (2*solver.dt) .* stor.dM[:] .+ stor.M0[:]
    lmul!(2*solver.dt, stor.dM) 
    stor.M2 .= stor.dM .+ stor.M0

    # M(t-dt) -> M(t)
    # M(t) -> M(t+dt)
    stor.M0 .= stor.M1
    stor.M1 .= stor.M2

    return nothing
end 

function propagate_RK2(m::AbstractVector{TR}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage; save=false) where TR
    # 574 μs without save and without checks, 630 μs with checks

    set_initial_state_2nd_order!(m, algebra, system, solver, stor)
    ts = collect(range(0.0, solver.tmax; step=solver.dt))
    n = length(ts)
    dmin = 1.0

    if save
        Us = Vector{typeof(stor.U)}(undef, n)
        Ms = Vector{typeof(stor.M0)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(stor.U0)
        Us[2] = copy(stor.U)
        Ms[1] = copy(stor.M0)
        Ms[2] = copy(stor.M1)
        dists[1] = distance(stor.U0, system, solver, stor)
        dists[2] = distance(stor.U, system, solver, stor)
    end

    for i in eachindex(ts)[3:end]

        propagator_2nd_order_step!(algebra, system, solver, stor)
        check_unitarity(stor.U, stor.tmp, timestep=i)
        @assert norm(stor.M0) < 2 "norm of M0: $(norm(stor.M0)) at timestep  $i"
        @assert norm(stor.M1) < 2 "norm of M1: $(norm(stor.M1)) at timestep  $i"
        @assert norm(stor.M2) < 2 "norm of M2: $(norm(stor.M2)) at timestep  $i"
        dist = distance(stor.U, system, solver, stor) 

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M0)  # instead of M1
            dists[i] = dist
        else
            if dist < solver.tol
                return dist
            elseif dist < dmin #&& 5 < ts[i]
                dmin = dist
            end
        end
    end
    return save ? (ts, Us, Ms, dists) : dmin
end


function RK4_step!(stor::Storage, algebra::Algebra, system::System, solver::SolverParams)

    # convert costate to Hilbert space 
    fill!(stor.M, zero(eltype(stor.M)))
    for μ in eachindex(stor.M_arr)
        stor.M .+= stor.M_arr[μ] .* algebra.lie_basis[μ]
    end 

    # H_opt(t) = argmax_H tr(H*M(t))
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M, algebra, system, solver, stor)

    # U(t+dt) = exp(H_opt(t) * dt) * U(t)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.dU, stor.tmp, stor.U)
    stor.U .= stor.dU

    # convert adjoint drift to lie coeffs
    project_to_algebra!(stor.adjoint_drift_arr, stor.adjoint_drift, algebra, stor)

    # k1 = [H_opt(M(t)), M(t)] in coeffs
    lie_bracket_coeffs!(stor.k1_arr, algebra.structure_tensor, stor.adjoint_drift_arr, stor.M_arr)

    # k2 = [H_opt(M(t) + k1*dt/2), M(t) + k1*dt/2]
    stor.tmp_M1_arr .= stor.M_arr .+ stor.k1_arr .* solver.dt ./ 2
    optimal_adjoint_drift!(stor.tmp_adj_drift1_arr, stor.tmp_M1_arr, algebra, system, solver, stor)
    lie_bracket_coeffs!(stor.k2_arr, algebra.structure_tensor, stor.tmp_adj_drift1_arr, stor.tmp_M1_arr)

    # k3 = [H_opt(M(t) + k2*dt/2), M(t) + k2*dt/2]
    stor.tmp_M2_arr .= stor.M_arr .+ stor.k2_arr .* solver.dt ./ 2
    optimal_adjoint_drift!(stor.tmp_adj_drift2_arr, stor.tmp_M2_arr, algebra, system, solver, stor)
    lie_bracket_coeffs!(stor.k3_arr, algebra.structure_tensor, stor.tmp_adj_drift2_arr, stor.tmp_M2_arr)

    # k4 = [H_opt(M(t) + k3*dt), M(t) + k3*dt]
    stor.tmp_M3_arr .= stor.M_arr .+ stor.k3_arr .* solver.dt 
    optimal_adjoint_drift!(stor.tmp_adj_drift3_arr, stor.tmp_M3_arr, algebra, system, solver, stor)
    lie_bracket_coeffs!(stor.k4_arr, algebra.structure_tensor, stor.tmp_adj_drift3_arr, stor.tmp_M3_arr)
    
    # M(t+dt) = M(t) + dt/6 * (k1 + 2k2 + 2k3 + k4)
    stor.M_arr .+= solver.dt / 6 .* (stor.k1_arr .+ 2 .* stor.k2_arr .+ 2 .* stor.k3_arr .+ stor.k4_arr)

    return nothing
end 

function propagate_RK4(m::AbstractVector{TR}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage; save=false) where TR

    build_M0!(stor.M, m, algebra)
    stor.M_arr[1]= 0.0
    stor.M_arr[2:end] .= m
    copyto!(stor.U, stor.U0)
    ts = collect(range(0.0, solver.tmax; step=solver.dt))
    n = length(ts)
    dmin = 1.0
    if save
        Us = Vector{typeof(stor.U)}(undef, n)
        Ms = Vector{typeof(stor.M)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(stor.U0)
        Ms[1] = copy(stor.M)
        dists[1] = distance(stor.U0, system, solver, stor)
    end

    for i in eachindex(ts)[2:end]

        RK4_step!(stor, algebra, system, solver)
        check_unitarity(stor.U, stor.tmp, timestep=i)
        dist = distance(stor.U, system, solver, stor) 

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M)
            dists[i] = dist
        else
            if dist < solver.tol
                return dist
            elseif dist < dmin
                dmin = dist
            end
        end
    end
    return save ? (ts, Us, Ms, dists) : dmin
end