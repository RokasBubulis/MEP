include("adjoint_drift_maximisation.jl")
include("distance.jl")

# function check_duals(x, name)
#     if eltype(x) <: ForwardDiff.Dual || eltype(x) <: Complex{<:ForwardDiff.Dual}
#         partials = ForwardDiff.partials.(real.(x))
#         if all(iszero, partials)
#             @warn "Dual parts are zero at $name"
#         end
#     end
# end

function build_M0!(M0::Matrix{TC}, m::AbstractVector{TR}, algebra::Algebra) where {TR, TC}
    M0 .= 0
    check_duals(m, "m")
    for (i, m_coeff) in enumerate(m)
        M0 .+= m_coeff * algebra.p_basis[i]
    end
    check_duals(M0, "M0")
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
    check_duals(stor.M0, "M0 initial set")
    optimal_adjoint_drift!(stor.adjoint_drift, stor.M0, algebra, system, solver, stor)  # H_opt(0)
    check_duals(stor.adjoint_drift, "Initial adjoint drift")
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
    check_duals(stor.adjoint_drift, "Adjoint drift")

    # U(t+dt) = exp(H_opt * dt) * U(t)
    exponent!(stor.tmp, stor.adjoint_drift * solver.dt)
    mul!(stor.dU, stor.tmp, stor.U)
    stor.U[:] .= stor.dU[:]
    check_duals(stor.U, "U")

    # dM/dt = [H_opt, M(t)]
    # mul!(stor.dM, stor.adjoint_drift, stor.M1)
    # mul!(stor.tmp, stor.M1, stor.adjoint_drift)
    # stor.dM[:] .-= stor.tmp[:]
    bracket_via_lie_coeffs!(stor.dM, stor.adjoint_drift, stor.M1, algebra, stor)
    check_duals(stor.dM, "dM")
    # project_to_algebra!(stor.tmp_array1, stor.adjoint_drift, algebra, stor)
    # project_to_algebra!(stor.tmp_array2, stor.M1, algebra, stor)
    # lie_bracket_coeffs!(stor.tmp_array3, algebra.structure_tensor, stor.tmp_array1, stor.tmp_array2)
    # fill!(stor.dM, zero(eltype(stor.dM)))
    # for μ in eachindex(stor.tmp_array3)
    #     stor.dM .+= stor.tmp_array3[μ] .* algebra.lie_basis[μ]
    # end 

    # M(t+dt) = 2*dt*[H_opt, M(t)] + M(t-dt)
    stor.M2[:] .= (2*solver.dt) .* stor.dM[:] .+ stor.M0[:]
    check_duals(stor.M2, "M2")

    # M(t-dt) -> M(t)
    # M(t) -> M(t+dt)
    # stor.M0[:] .= stor.M1[:]
    # stor.M1[:] .= stor.M2[:]
    stor.M0 .= stor.M1
    stor.M1 .= stor.M2

    return nothing
end 

function propagate(m::AbstractVector{TR}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage; save=false) where TR
    # 574 μs without save and without checks, 630 μs with checks

    set_initial_state_2nd_order!(m, algebra, system, solver, stor)
    ts = collect(range(0.0, solver.tmax; step=solver.dt))
    n = length(ts)

    d1 = distance(stor.U0, system, solver, stor)
    d2 = distance(stor.U, system, solver, stor)
    dmin = min(d1, d2)
    #check_duals(d1, "d1")
    check_duals(stor.U, "U(step 1)")
    check_duals(d2, "d2")

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

        check_belongs_to_p_subspace(stor.adjoint_drift, algebra; timestep=i, identifier="Optimal adjoint drift")
        check_belongs_to_p_subspace(stor.M1, algebra; timestep=i, identifier="Costate")
        propagator_2nd_order_step!(algebra, system, solver, stor)
        check_belongs_to_p_subspace(stor.dM, algebra; timestep=i, identifier="Costate differential")
        check_unitarity(stor.U, stor.tmp, timestep=i)
        @assert norm(stor.M0) < 2 "norm of M0: $(norm(stor.M0)) at timestep  $i"
        @assert norm(stor.M1) < 2 "norm of M1: $(norm(stor.M1)) at timestep  $i"
        @assert norm(stor.M2) < 2 "norm of M2: $(norm(stor.M2)) at timestep  $i"
        dist = distance(stor.U, system, solver, stor)

        check_duals(stor.M0, "M(t)")
        check_duals(stor.U, "U(t)")
        check_duals(dist, "dmin(t)")

        if save
            Us[i] = copy(stor.U)
            Ms[i] = copy(stor.M0)  # instead of M1
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
