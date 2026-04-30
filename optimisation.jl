using ForwardDiff, CMAEvolutionStrategy

include("propagation.jl")
include("checks.jl")

### Shooting approach to find the best initial costate M0 ###

# represent the search space as n-1 angles on the hyper-sphere ensuring |M(t)| = 1
# function angles_to_directions(angles::AbstractVector{T}) where T
#     n = length(angles) + 1
#     m = Vector{T}(undef, n)
#     m[1] = cos(angles[1])
#     prefix = sin(angles[1])
#     for i in 2:n-1
#         m[i] = prefix * cos(angles[i])
#         prefix *= sin(angles[i])
#     end
#     m[n] = prefix
#     check_duals(m, "m(θ)")
#     return m
# end

function angles_to_directions(v::AbstractVector{T}) where T
    n = norm(v)
    if n < eps(real(T))
        return v ./ one(real(T))  # fallback, shouldn't happen
    end
    return v ./ n
end

function find_best_initial_costate_bbf(algebra::Algebra, system::System, solver::SolverParams, stor::Storage)

    # check target before propagation
    check_unitarity(system.target, stor.tmp, note="Target")
    targ_dist = distance(system.target, system, solver,stor)
    @assert targ_dist < solver.tol "Error in target overlap: $targ_dist"

    n = length(algebra.p_basis) - 1
    initial_angles = zeros(n)

    thread_stors = [deepcopy(stor) for _ in 1:Threads.maxthreadid()]

    objective = function(angles)
        s = thread_stors[Threads.threadid()]
        m = angles_to_directions(angles)
        list = [0, 1, 2, 2, 3, 4, 4]
        propagate_2nd_order(m, algebra, system, solver, s) + solver.lambda *  dot(list, abs.(m).^2)
    end

    # use CMA Evolution strategy for optimisation
    result = minimize(
        objective,
        initial_angles, 1.0;
        maxiter=500, verbosity=1,
        multi_threading=true,
        popsize=24,
        lower = zeros(n),
        upper = [i == 1 ? 2π : π for i in 1:n],
        ftarget=solver.lambda != 0 ? solver.lambda * solver.tol : solver.tol
        )
    angles_best = xbest(result)
    m_best = angles_to_directions(angles_best)

    return m_best
end


function find_best_initial_costate_autograd(algebra::Algebra, system::System, solver::SolverParams, stor::Storage; verbose = true)

    # check target before propagation
    check_unitarity(system.target, stor.tmp, note="Target")
    targ_dist = distance(system.target, system, solver, stor)
    @assert targ_dist < solver.tol "Error in target overlap: $targ_dist"

    # n-hypersphere angles
    # n = length(algebra.p_basis) - 1
    # x0 = zeros(n)

    # n independent directions
    x0 = zeros(length(algebra.p_basis))
    x0[1] = 1.0

    dim = size(stor.adjoint_drift, 1)
    stor_dual_ref = Ref{Any}(nothing)
    objective = function(angles)
        m = angles_to_directions(angles)
        if eltype(m) <: ForwardDiff.Dual
            T = Complex{eltype(m)}
            if typeof(stor_dual_ref[]) != Storage{T}
                stor_dual_ref[] = Storage{T}(dim)
            end
            storage = stor_dual_ref[]
        else
            storage = stor
        end
        propagate_2nd_order(m, algebra, system, solver, storage)
    end

    g! = (G, x) -> ForwardDiff.gradient!(G, objective, x)
    od = OnceDifferentiable(objective, g!, x0)

    # Test that ForwardDiff can actually differentiate your objective
    grad = ForwardDiff.gradient(objective, x0)
    @assert any(!iszero, grad) "Initial gradient is all zeros"
    # od = OnceDifferentiable(objective, initial_angles; autodiff = :forward)
    angles_best = Optim.minimizer(
        optimize(
            od, x0, 
            BFGS(linesearch = BackTracking()), 
            Optim.Options(
                show_trace=verbose, 
                f_abstol=solver.tol,
                g_tol=1e-12,
                )
        )
    )

    m_best = angles_to_directions(angles_best)

    return m_best
end

