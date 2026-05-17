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

    n = length(algebra.p_basis)
    m0 = zeros(n)
    m0[1] = 1.0

    objective = function(m)
        m ./= norm(m)
        propagate(m, algebra, system, solver, stor)
    end

    result = Optim.optimize(objective, m0, NelderMead(), Optim.Options(
        show_trace  = true,   # print iteration log
        #extended_trace = true, # include simplex details
        iterations  = 100,
        f_abstol = solver.tol,
    ))
    m_best = result.minimizer

    return m_best
end


function find_best_initial_costate_autograd(algebra::Algebra, system::System, solver::SolverParams, stor::Storage; verbose = true)

    # check target before propagation
    check_unitarity(system.target, stor.tmp, note="Target")
    targ_dist = distance(system.target, system, solver, stor)
    @assert targ_dist < solver.tol "Error in target overlap: $targ_dist"

    # n independent directions
    x0 = zeros(length(algebra.p_basis))
    x0[1] = 1.0  # setting a different initial direction causes either diverging gradients or operators not in lie basis

    dim = size(stor.adjoint_drift, 1)
    stor_dual_ref = Ref{Any}(nothing)
    objective = function(m)
        #m = angles_to_directions(angles)
        if eltype(m) <: ForwardDiff.Dual
            T = Complex{eltype(m)}
            if typeof(stor_dual_ref[]) != Storage{T}
                stor_dual_ref[] = Storage{T}(dim, length(algebra.lie_basis))
            end
            storage = stor_dual_ref[]
        else
            storage = stor
        end
        propagate(m, algebra, system, solver, storage)
    end

    g! = (G, x) -> ForwardDiff.gradient!(G, objective, x)
    od = OnceDifferentiable(objective, g!, x0)

    # Test that ForwardDiff can actually differentiate your objective
    # grad = ForwardDiff.gradient(objective, x0)
    # @assert any(!iszero, grad) "Initial gradient is all zeros"
    # od = OnceDifferentiable(objective, initial_angles; autodiff = :forward)
    m_best = Optim.minimizer(
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

    # m_best = angles_to_directions(angles_best)

    return m_best
end

