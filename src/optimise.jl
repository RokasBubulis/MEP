using Optim
using NLSolversBase: only_fg!

include("new_propagation.jl")

function make_optimisation_problem(algebra::Algebra, solver::SolverParams, tar::TargetContainer)
    return function (F, G, x)
        tracker = new_propagate(x, algebra, solver, tar)
        G === nothing || grad_infidelity!(G, algebra, stor)
        F === nothing || return infidelity(stor.U, tar, algebra, stor)
        return nothing
    end 
end 

function run_optimiser(tar::TargetContainer, algebra::Algebra, x0::Vector{Float64})

    # wrap cost and gradient functions
    cg! = make_optimisation_problem(tar, algebra)

    opts = Optim.Options(; iterations = 1000, g_tol = 1e-8, show_trace = false)
    res = optimize(only_fg!(cg!), x0, LBFGS(), opts)

    return Optim.minimizer(res), Optim.minimum(res), res
end 



##
# EXAMPLE 
"""
    optimise(cost, grad!, x0; method=LBFGS(), iterations=1000, g_tol=1e-8, show_trace=false)

Minimise `cost(x)` given its analytic gradient `grad!(G, x)`, which must fill `G` in place.
Returns `(minimiser, minimum, result)`.
"""
function optimise(cost, grad!, x0::AbstractArray;
                  method = LBFGS(), iterations = 1000, g_tol = 1e-8,
                  show_trace = false)
    function fg!(F, G, x)
        G === nothing || grad!(G, x)
        F === nothing || return cost(x)
        return nothing
    end
    opts = Optim.Options(; iterations = iterations, g_tol = g_tol,
                           show_trace = show_trace)
    res = optimize(only_fg!(fg!), float.(x0), method, opts)
    return Optim.minimizer(res), Optim.minimum(res), res
end

"""
    check_gradient(cost, grad!, x; h=1e-6)

Max absolute difference between `grad!` and a central finite difference.
Expect roughly 1e-6 or smaller if the gradient is correct.
"""
function check_gradient(cost, grad!, x::AbstractArray; h = 1e-6)
    x = float.(x)
    g = similar(x); grad!(g, x)
    gfd = similar(x)
    for i in eachindex(x)
        xp = copy(x); xm = copy(x)
        xp[i] += h;   xm[i] -= h
        gfd[i] = (cost(xp) - cost(xm)) / (2h)
    end
    return maximum(abs.(g .- gfd))
end
##

rosen(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
function rosen_grad!(G, x)
    G[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
    G[2] = 200 * (x[2] - x[1]^2)
end

x0 = [-1.2, 1.0]
println("gradient check: ", check_gradient(rosen, rosen_grad!, x0))
xopt, fmin, res = optimise(rosen, rosen_grad!, x0)
println("minimiser: ", xopt, "  minimum: ", fmin)
println("converged: ", Optim.converged(res),
        "  iterations: ", Optim.iterations(res))