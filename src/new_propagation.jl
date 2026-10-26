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

    # costate sensitivity propagation
    # dχ/dt = χD in Lie space; dim = dxd
    costate_sensitivity_drift!(stor.D, algebra, stor, M)
    mul!(dχ, χ, stor.D)

    # state sensitivity propagation
    # obtain E 
    state_sensitivity_source!(stor.E, stor)
    # obtain F 
    state_sensitivity_drift!(stor.F, algebra, stor)
    # dB/dt = χE + BF 
    mul!(dB, χ, stor.E)
    mul!(dB, B, stor.F, 1, 1)  # mul!(A,B,C,a,b) = A -> a*B*C + b*A
end 

function manual_unitary_propagation!(M_lie::Vector{Float64}, integrator, algebra::Algebra, stor::Storage)
    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, M_lie, algebra, stor)
    Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
    copyto!(stor.dU, stor.H_opt)
    lmul!(integrator.dt, stor.dU)
    exponential!(stor.dU, stor.exp_method, stor.exp_cache)
    mul!(stor.U_buffer, stor.dU, stor.U)
    stor.U .= stor.U_buffer  
end 

function manual_optimum_update!(u, t, algebra::Algebra, stor::Storage, tar::TargetContainer, tracker::CostOptimumTracker)
    cost = infidelity(stor.U, algebra, stor, tar)
    # TODO investigate whether the cost should include the running time
    if cost < tracker.min_cost && t > 5
        _, _, B = u.x
        tracker.min_cost = cost 
        tracker.t_star = t
        tracker.B_t_star .= B
        tracker.U_t_star .= stor.U
        tracker.W_t_star .= conj(exp(stor.β_opt * Matrix(algebra.lie_basis[1])) * tar.target)  # TODO add allocation-free implementation
    end
end 

function unconditional_manual_optimum_update!(u, t, algebra::Algebra, stor::Storage, tar::TargetContainer, tracker::CostOptimumTracker)
    cost = infidelity(stor.U, algebra, stor, tar)
    # TODO investigate whether the cost should include the running time
    M, χ, B = u.x
    tracker.min_cost = cost 
    tracker.t_star = t
    tracker.B_t_star .= B
    tracker.U_t_star .= stor.U
    tracker.W_t_star .= conj(exp(stor.β_opt * Matrix(algebra.lie_basis[1])) * tar.target)  # 
    tracker.M_t_star .= M
    tracker.χ_t_star .= χ
    # TODO add allocation-free implementation
end 

function new_propagate(m0::Vector{Float64}, algebra::Algebra, solver::SolverParams, tar::TargetContainer; dt=1e-2, method=Midpoint())

    # initialisation
    d = length(algebra.lie_basis)
    n = algebra.n_levels^algebra.n_particles
    stor = Storage(algebra.n_particles, d)
    # m0 = zeros(Float64, d)
    # m0[2:end] = m
    stor.U .= stor.U0 
    x0 = ArrayPartition(m0, Matrix{Float64}(I, d, d), zeros(Float64, d, d))

    # setup optimal point tracker
    tracker = CostOptimumTracker(1.0, 0.0, Matrix{Float64}(undef, d, d), Matrix{ComplexF64}(undef, n, n), Matrix{ComplexF64}(undef, n, n), Vector{Float64}(undef, d), Matrix{Float64}( undef, d, d))

    # setup ODE problem
    params = algebra, stor, tracker, tar, solver
    prob = ODEProblem(ode_problem!, x0, (0.0, solver.tmax), params)

    # callback 
    step_cb = FunctionCallingCallback(func_everystep=true) do u, t, integrator
        M, _, _ = u.x
        manual_unitary_propagation!(M, integrator, algebra, stor)
        #manual_optimum_update!(u, t, algebra, stor, tar, tracker)

    end 

    # solve 
    sol = solve(prob, method; callback=step_cb, reltol=solver.reltol, abstol=solver.abstol, adaptive=false, dt=dt, save_everystep=false, dense=false)

    unconditional_manual_optimum_update!(sol.u[end], sol.t[end], algebra, stor, tar, tracker)

    return tracker
end 