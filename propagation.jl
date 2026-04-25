include("adjoint_drift_maximisation.jl")
include("distance.jl")

function check_duals(x, name)
    if eltype(x) <: ForwardDiff.Dual || eltype(x) <: Complex{<:ForwardDiff.Dual}
        partials = ForwardDiff.partials.(real.(x))
        if all(iszero, partials)
            @warn "Dual parts are zero at $name"
        end
    end
end

function build_M0!(M0::Matrix{TC}, m::AbstractVector{TR}, algebra::Algebra) where {TR, TC}
    M0 .= 0
    check_duals(m, "m")
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * algebra.p_basis[i]
    end
    check_duals(M0, "M0")
    return nothing
end 

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

function set_initial_state_2nd_order!(m::AbstractVector{TR}, algebra::Algebra, system::System, 
    solver::SolverParams, stor::Storage) where TR
    # set U_tmp = U(dt) = exp(H_opt(0)*dt) * U(0) and 
    # M_tmp0 = M(0), M_tmp1 = M(dt)
    # to build M(dt) for the first step, use first-order approximation

    build_M0!(stor.M0, m, algebra)  # M(0)
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M0, system, solver)  # H_opt(0)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.U, stor.tmp, stor.U0)  # U(dt)

    # M(dt) = [H_opt(0), M(0)] * dt
    mul!(stor.dM, stor.adjoint_drift, stor.M0)
    mul!(stor.tmp, stor.M0, stor.adjoint_drift)
    stor.dM .-= stor.tmp
    stor.M1 .= stor.dM * solver.dt .+ stor.M0

    return nothing
end

function propagator_2nd_order_step!(system::System, solver::SolverParams, stor::Storage)

    # compute H_opt(t) = argmax_H(α) tr(H(α)*M(t))
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M1, system, solver)

    # U(t+dt) = exp(H_opt * dt) * U(t)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.dU, stor.tmp, stor.U)
    stor.U[:] .= stor.dU[:]

    # dM/dt = [H_opt, M(t)]
    mul!(stor.dM, stor.adjoint_drift, stor.M1)
    mul!(stor.tmp, stor.M1, stor.adjoint_drift)
    stor.dM[:] .-= stor.tmp[:]

    # M(t+dt) = 2*dt*[H_opt, M(t)] + M(t-dt)
    stor.M2[:] .= (2*solver.dt) .* stor.dM[:] .+ stor.M0[:]

    # M(t-dt) -> M(t)
    # M(t) -> M(t+dt)
    stor.M0[:] .= stor.M1[:]
    stor.M1[:] .= stor.M2[:]

    return nothing
end 

function propagate_2nd_order(m::AbstractVector{TR}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage; save=false) where TR
    # 574 μs without save and without checks, 630 μs with checks

    set_initial_state_2nd_order!(m, algebra, system, solver, stor)
    ts = collect(range(0.0, solver.tmax; step=solver.dt))
    n = length(ts)

    d1 = distance(stor.U0, system, solver, stor)
    d2 = distance(stor.U, system, solver, stor)
    dmin = min(d1, d2)
    check_duals(stor.U, "U(dt)")

    if save
        Us = Vector{typeof(stor.U)}(undef, n)
        Ms = Vector{typeof(stor.M0)}(undef, n)
        dists = Vector{Float64}(undef, n)
        Us[1] = copy(stor.U0)
        Us[2] = copy(stor.U)
        Ms[1] = copy(stor.M0)
        Ms[2] = copy(stor.M1)
        dists[1] = d1
        dists[2] = d2
    end

    for i in eachindex(ts)[3:end]

        check_unitarity(stor.U, stor.tmp, timestep=i)
        check_belongs_to_p_subspace(stor.dM, algebra; timestep=i, identifier="Costate differential")
        check_belongs_to_p_subspace(stor.M0, algebra; timestep=i, identifier="Costate")
        propagator_2nd_order_step!(system, solver, stor)
        dist = distance(stor.U, system, solver, stor)

        check_duals(stor.M0, "M(t)")
        check_duals(stor.U, "U(t)")
        check_duals(dist, "dmin(t)")

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M1)
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
