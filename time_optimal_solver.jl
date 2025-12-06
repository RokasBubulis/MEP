using DifferentialEquations, Optim, FastExpm, ComponentArrays

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

br(A, B) = A*B - B*A

function obtain_H_opt(M, p)
    H0, L = p.H0, p.L

    function cost(alpha)
        alpha = alpha[1]
        return -real(tr(fastExpm(-alpha * L) * H0 * fastExpm(alpha * L) * M))
    end

    function cost_grad!(G, alpha)
        alpha = alpha[1]
        A = fastExpm(-alpha * L) * H0 * fastExpm(alpha * L)
        commutator = br(A, L)
        G[1] = real(tr(commutator * M))
    end

    alpha0 = [1.0]
    result = optimize(cost, cost_grad!, alpha0, LBFGS(); inplace=true)
    alpha_opt = Optim.minimizer(result)

    return -cost(alpha_opt)
end

function get_distance_to_target_coset(P_opt, target, p)
    L = p.L
    target_P_product = adjoint(target)*P_opt
    function cost(alpha)
        alpha = alpha[1]
        return norm(target_P_product - fastExpm(alpha * L))
    end

    function cost_grad!(G, alpha)
        alpha = alpha[1]
        G[1] = norm(-L*fastExpm(alpha * L))
    end
    
    alpha0 = [1.0]
    result = optimize(cost, cost_grad!, alpha0, LBFGS(); inplace=true)
    alpha_opt = Optim.minimizer(result)[1]

    return norm(target_P_product - fastExpm(alpha_opt*L))
end

# Coupled ODE system, (P_opt, M)
# dP_opt/dt = H_opt(t) * P_opt
# dM/dt = [M, H_opt(t)]
function f!(dX, X, p, t)
    P, M = X.P, X.M
    H_opt = obtain_H_opt(M, p)
    dX.P .= H_opt * P
    dX.M .= br(M, H_opt)
end


function compute_optimal_time(target, gens, M0; tmax = 10.0, coset_tolerance = 1e-3)
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable given the generators"

    X0 = ComponentArray(P = kron(id, id), M = M0)
    tspan = (0.0, tmax)
    p = ComponentArray(H0 = -im*gens[1], L = -im*gens[2], tol = coset_tolerance)

    prob = ODEProblem(f!, X0, tspan, p)

    condition(u, t, integrator) = true 

    function affect!(integrator)
        X = integrator.u
        P, M = X.P, X.M
        p = integrator.p

        dist = get_distance_to_target_coset(P, target, p)
        if dist < real(p.tol)
            println("Condition met at t = ", integrator.t)
            terminate!(integrator)
        end
    end

    cb = DiscreteCallback(condition, affect!)

    sol = solve(prob, callback=cb)
    println("Termination status: ", sol.retcode)
    if sol.t[end] == tspan[end]
        println("Target coset not reached within tmax = $(sol.t[end])")
    else
        println("Final time: ", sol.t[end])
    end

end