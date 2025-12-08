using DifferentialEquations, Optim, FastExpm, ComponentArrays

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

br(A, B) = A*B - B*A

@inline function H_of_alpha!(H_of_alpha, H0, Delta, alpha)
    @inbounds @simd for k in eachindex(H0)
        H_of_alpha[k] = H0[k] * exp(alpha * Delta[k])
    end
end

@inline function commutator_diagL!(C, H_of_alpha, Delta)
    @inbounds @simd for k in eachindex(H_of_alpha)
        C[k] = -Delta[k] * H_of_alpha[k]
    end
end

function obtain_H_opt(M, p)
    H0, Delta = p.H0, p.Delta
    H_of_alpha = similar(H0)
    C = similar(H0)

    function cost(alpha)
        alpha = alpha[1]
        # return -real(tr(fastExpm(-alpha * L) * H0 * fastExpm(alpha * L) * M))
        H_of_alpha!(H_of_alpha, H0, Delta, alpha)
        return -real(tr(H_of_alpha* M))
    end

    function cost_grad!(G, alpha)
        alpha = alpha[1]
        H_of_alpha!(H_of_alpha, H0, Delta, alpha)
        commutator_diagL!(C, H_of_alpha, Delta)
        # A = fastExpm(-alpha * L) * H0 * fastExpm(alpha * L)
        # commutator = br(H(alpha), L)
        G[1] = real(tr(C * M))
    end

    alpha0 = [1.0]
    result = optimize(cost, cost_grad!, alpha0, LBFGS(); inplace=true)
    alpha_opt = Optim.minimizer(result)
    H_of_alpha!(H_of_alpha, H0, Delta, alpha_opt[])

    return H_of_alpha
end

# function get_distance_to_target_coset(P_opt, target, p)
#     L, diag_L = p.L, p.diag_L
#     target_P_product = adjoint(target)*P_opt
#     function cost(alpha)
#         alpha = alpha[1]
#         return norm(target_P_product - exp_alpha_L(alpha, diag_L))
#     end

#     function cost_grad!(G, alpha)
#         alpha = alpha[1]
#         G[1] = norm(-L*exp_alpha_L(alpha, diag_L))
#     end
    
#     alpha0 = [1.0]
#     result = optimize(cost, cost_grad!, alpha0, LBFGS(); inplace=true)
#     alpha_opt = Optim.minimizer(result)[1]

#     return norm(target_P_product - exp_alpha_L(alpha_opt, diag_L))
# end

function get_distance_to_target_coset(P_opt, target, p, alpha_memory)
    λ = p.diag_L
    A = adjoint(target) * P_opt

    function cost(alpha)
        alpha = alpha[1]
        return norm(A - spdiagm(0 => exp.(alpha .* λ)))
    end

    result = optimize(cost, [alpha_memory[]], LBFGS(); inplace=true)
    alpha_opt = Optim.minimizer(result)[1]

    alpha_memory[] = alpha_opt
    return cost([alpha_opt])
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

function build_M(m, p_basis)
    M = similar(p_basis[1])
    for (i,m_coeff) in enumerate(m)
        M .+= m_coeff * p_basis[i]
    end   
    return M
end

function compute_optimal_time(target, gens; tmax = 10.0, coset_tolerance = 1e-3, penalty = 10.0)

    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = [[lie_basis[1]]; lie_basis[3:end]] # assumption that controls are given by basis element number 2 (after GSO)

    L = -im*gens[2]
    diag_L = diag(L)
    Delta = diag_L' .- diag_L
    p = ComponentArray(H0 = -im*gens[1], Delta = Delta, diag_L = diag_L, tol = coset_tolerance) #is minus causing inconsistencies with lie algebra?

    function single_sol_run(m; tmax=tmax)

        M0 = build_M(m, p_basis)
        X0 = ComponentArray(P = kron(id, id), M = M0)
        tspan = (0.0, tmax)
        prob = ODEProblem(f!, X0, tspan, p)

        min_dist = Ref(Inf)
        hit_time = Ref(nothing)
        alpha_memory = Ref(0.0)
        condition(u, t, integrator) = (integrator.iter % 10 == 0)

        function affect!(integrator)
            X = integrator.u
            P = X.P
            p = integrator.p

            dist = get_distance_to_target_coset(P, target, p, alpha_memory)
            if dist < min_dist[]
                min_dist[] = dist
            end 

            if dist < real(p.tol) && hit_time[] === nothing
                hit_time[] = integrator.t
                terminate!(integrator)
            end
        end

        cb = DiscreteCallback(condition, affect!)
        sol = solve(prob, callback=cb)

        return hit_time[], min_dist[]
    end

    function objective(m)
        hit_time, min_dist = single_sol_run(m)
        println("Evaluating at m = $m : hit_T = $hit_time, min_dist = $min_dist")
        if hit_time !== nothing
            return hit_time
        else
            return tmax + penalty * min_dist
        end
    end

    m0 = ones(length(p_basis))
    result = optimize(objective, m0, LBFGS())
    m_opt = Optim.minimizer(result)
    T_opt = Optim.minimum(result)
    println(result)
end