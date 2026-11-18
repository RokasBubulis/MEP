using DifferentialEquations, DiffEqCallbacks, OrdinaryDiffEqLowOrderRK, StaticArrays, RecursiveArrayTools

include("adm.jl")
include("sensitivity.jl")

function ode_problem!(du, u, p, t)
    M, χ, B = u.x
    dM, dχ, dB = du.x

    algebra, stor = p[1], p[2]
    # costate propagation
    # obtain H_opt 
    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, M, algebra, stor)
    # dM/dt = [H_opt, M] in Lie space; dim = dx1
    lie_bracket_coeffs!(dM, algebra.structure_tensor, stor.H_opt_lie, M)

    # obtain F (computed first: it is also the second term of D)
    state_sensitivity_drift!(stor.F, algebra, stor)

    # costate sensitivity propagation
    # dχ/dt = χD in Lie space; dim = dxd
    costate_sensitivity_drift!(stor.D, algebra, stor, M)
    mul!(dχ, χ, stor.D)

    # state sensitivity propagation
    # obtain E 
    state_sensitivity_source!(stor.E, stor)
    # dB/dt = χE + BF 
    mul!(dB, χ, stor.E)
    mul!(dB, B, stor.F, 1, 1)  # mul!(A,B,C,a,b) = A -> a*B*C + b*A
end 

function manual_unitary_propagation!(M_lie::Vector{Float64}, integrator, algebra::Algebra, stor::Storage; first_step=false)

    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, M_lie, algebra, stor)
    if first_step[]
        # no interval has been traversed yet, so there is nothing to advance:
        # just record H(t0) as the left endpoint of the first step
        copyto!(stor.H_opt_t_minus_dt_lie, stor.H_opt_lie)
        first_step[] = false
        return nothing
    end

    # midpoint propagation on every invariant subspace separately, through the isometries:
    # U^(λ)_{k+1} = exp(dt/2 (H^(λ)(t_k) + H^(λ)(t_{k+1}))) U^(λ)_k ,
    # H^(λ) = V_λ' H V_λ with V_λ the 3^N x 2 isometry spanning the λ-th invariant subspace
    Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
    Lie_to_Hilbert!(stor.H_opt_t_minus_dt, stor.H_opt_t_minus_dt_lie, algebra)
    @inbounds for λ in 1:algebra.n_particles
        apply_λth_isometry_transformation!(stor.H_opt_invariant[λ], stor.H_opt, algebra.isometries[λ], stor.tmp_for_isometry)
        apply_λth_isometry_transformation!(stor.H_opt_t_minus_dt_invariant[λ], stor.H_opt_t_minus_dt, algebra.isometries[λ], stor.tmp_for_isometry)
        copyto!(stor.dU[λ], stor.H_opt_invariant[λ])
        axpy!(1.0, stor.H_opt_t_minus_dt_invariant[λ], stor.dU[λ])
        lmul!(1/2, stor.dU[λ])
        lmul!(integrator.dt, stor.dU[λ])
        exponential!(stor.dU[λ], stor.exp_method, stor.exp_cache)
        mul!(stor.U_buffer[λ], stor.dU[λ], stor.U[λ])
        copyto!(stor.U[λ], stor.U_buffer[λ])
    end

    # advance the left endpoint of the next interval
    copyto!(stor.H_opt_t_minus_dt_lie, stor.H_opt_lie)
    return nothing
end 

# track_min = true : tracker holds the minimum cost over the trajectory on [0, T]
# track_min = false: tracker holds the cost at the final time T (smooth in m0 and T)
# nsteps given     : dt = T/nsteps, so the step grid scales with T
# cost_history     : if a Vector{Tuple{Float64,Float64}} is given, (t, φ(t)) is pushed at every step
# alpha_history    : if a Vector{Tuple{Float64,Float64}} is given, (t, α_opt(t)) is pushed at every step (incl. t = 0)
function propagate(m0::Vector{Float64}, algebra::Algebra, solver::SolverParams, tar::TargetContainer; dt=1e-2, method=Midpoint(), T=solver.tmax, nsteps=nothing, track_min=false, time_weight=0.0, cost_history=nothing, alpha_history=nothing)
    dt = nsteps === nothing ? dt : T/nsteps

    # initialisation
    d = length(algebra.lie_basis)
    N = algebra.n_particles
    stor = Storage(N, d)
    @inbounds for λ in 1:N
        copyto!(stor.U[λ], Matrix{ComplexF64}(I, 2, 2))
    end
    target_amplitudes!(stor, algebra, tar)
    x0 = ArrayPartition(m0, Matrix{Float64}(I, d, d), zeros(Float64, d, d))

    params = algebra, stor
    prob = ODEProblem(ode_problem!, x0, (0.0, T), params)

    # setup optimal point tracker
    full_dim = 3^N
    tracker = PropagationOutput(Inf, 0.0, [Matrix{ComplexF64}(I, 2, 2) for _ in 1:N],
                                zeros(Float64, d, d), copy(m0), Matrix{Float64}(I, d, d),
                                copy(stor.H_opt_lie), zeros(ComplexF64, N + 1), copy(stor.target_amplitude),
                                zero(Float64), tar.k, tar.ϕ,
                                Matrix{ComplexF64}(undef, full_dim, full_dim),
                                Matrix{ComplexF64}(undef, 2, 2),
                                Matrix{ComplexF64}(undef, 2, full_dim))

    update_tracker!(tracker, stor, sol_t=nothing) = nothing
    function save_state!(tracker, stor, u, t)
        M, χ, B = u.x
        tracker.t_star = t
        copyto!(tracker.B_t_star, B)
        @inbounds for λ in 1:N
            copyto!(tracker.U_t_star[λ], stor.U[λ])
        end
        copyto!(tracker.M_t_star, M)
        copyto!(tracker.χ_t_star, χ)
        copyto!(tracker.H_t_star, stor.H_opt_lie)
        tracker.a_t .= state_diag(tracker.U_t_star)
        tracker.phase_opt = stor.phase_opt
        return nothing
    end

    # The callback fires once per accepted step *and* once at t = 0 (func_start),
    # i.e. T/dt + 1 times. manual_unitary_propagation! consumes first_step
    # itself so that U is advanced over exactly the T/dt intervals the solver
    # traverses, leaving U, M, χ and B all evaluated at t = T.
    first_step = Ref(true)
    step_cb = FunctionCallingCallback(func_everystep=true) do u, t, integrator
        M, _, _ = u.x
        manual_unitary_propagation!(M, integrator, algebra, stor; first_step=first_step)
        if cost_history !== nothing
            φ = cost(stor.U, t, algebra, stor, tar.k, tar.ϕ, 0.0)
            push!(cost_history, (t, φ))
        end
        if track_min
            φ = cost(stor.U, t, algebra, stor, tar.k, tar.ϕ, 0.0)
            if φ < tracker.min_cost
                tracker.min_cost = φ
                save_state!(tracker, stor, u, t)
            end
        end
        alpha_history === nothing || push!(alpha_history, (t, stor.alpha))
    end

    # solve
    sol = solve(prob, method; callback=step_cb, reltol=solver.reltol, abstol=solver.abstol, adaptive=false, dt=dt, save_everystep=false, dense=false)

    # wrap output: state of the propagation at t = T
    φ = cost(stor.U, T, algebra, stor, tar.k, tar.ϕ, 0.0)
    if !track_min
        save_state!(tracker, stor, sol.u[end], T)
        tracker.min_cost = φ
    end
    tracker.min_cost += time_weight * T

    return tracker
end 
