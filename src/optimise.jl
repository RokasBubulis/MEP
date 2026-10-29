using Optim
using NLSolversBase: only_fg!

# T = tmin + (tmax - tmin) σ(τ) keeps tmin < T < tmax while the optimiser works unconstrained in τ
# (tmin = solver.tstar_min, tmax = solver.tstar_max; tmin > 0 excludes the trivial T → 0, U = I minimum)
T_of_τ(τ, tmin, tmax) = tmin + (tmax - tmin) / (1 + exp(-τ))
τ_of_T(T, tmin, tmax) = log((T - tmin) / (tmax - T))

# x = [m; τ]: m is the initial costate direction (only m/|m| matters), T = T_of_τ(τ) the final time
# J(m, T) = φ(U(T; m)) + time_weight * T  is smooth in both m and T
function make_optimisation_problem(algebra::Algebra, solver::SolverParams, tar::TargetContainer; time_weight=1e-3, nsteps=1000)
    d = length(algebra.lie_basis)
    grad_m = zeros(Float64, d)
    return function (F, G, x)
        r = norm(view(x, 1:d))
        m = x[1:d] ./ r
        T = T_of_τ(x[end], solver.tstar_min, solver.tstar_max)
        tracker = new_propagate(m, algebra, solver, tar; T=T, nsteps=nsteps, track_min=false)
        if G !== nothing
            grad_infidelity!(grad_m, algebra, tracker)
            grad_m .-= dot(grad_m, m) .* m      # φ is invariant under m → λm: keep the tangent part
            G[1:d] .= grad_m ./ r
            tmin, tmax = solver.tstar_min, solver.tstar_max
            G[end] = (dphi_dT(tracker) + time_weight) * (T - tmin) * (tmax - T) / (tmax - tmin)   # dT/dτ
        end
        F === nothing || return tracker.min_cost + time_weight * T
        return nothing
    end
end

# (m, T) from an optimiser result on x = [m; τ]
function minimizer_mT(res, solver::SolverParams)
    x = Optim.minimizer(res)
    return normalize(x[1:end-1]), T_of_τ(x[end], solver.tstar_min, solver.tstar_max)
end

# g_tol: the analytic gradient matches the discretised cost only to O(dt²) (≈1e-4 at dt = 1e-2),
# so smaller gradient tolerances cannot be met and BFGS would run to the iteration limit
function run_bfgs(m0::Vector{Float64}, T0::Float64, algebra::Algebra, solver::SolverParams, tar::TargetContainer; time_weight=1e-3, nsteps=1000, iterations=100, g_tol=1e-4, f_reltol=1e-8)

    # wrap cost and gradient functions
    cg! = make_optimisation_problem(algebra, solver, tar; time_weight=time_weight, nsteps=nsteps)

    d = length(algebra.lie_basis)
    x0 = [normalize(m0); τ_of_T(T0, solver.tstar_min, solver.tstar_max)]

    opts = Optim.Options(; iterations = iterations, g_tol = g_tol, f_reltol = f_reltol, show_trace = true)
    res = optimize(only_fg!(cg!), x0, BFGS(), opts)

    return res
end

function run_nelder_mead(m0::Vector{Float64}, T0::Float64, algebra::Algebra, solver::SolverParams, tar::TargetContainer; time_weight=1e-3, nsteps=1000, iterations=1000)

    # value-only call of the cost used by run_bfgs
    cg! = make_optimisation_problem(algebra, solver, tar; time_weight=time_weight, nsteps=nsteps)
    J(x) = cg!(true, nothing, x)

    x0 = [normalize(m0); τ_of_T(T0, solver.tstar_min, solver.tstar_max)]

    opts = Optim.Options(; iterations = iterations, g_tol = 1e-8, show_trace = true)
    res = optimize(J, x0, NelderMead(), opts)

    return res
end
