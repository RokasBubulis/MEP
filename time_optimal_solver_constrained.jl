using DifferentialEquations, Optim, FastExpm, ComponentArrays, Roots

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

mutable struct Params{T1,T2,T3}
    H0::T1
    l::T2
    tol::Float64  # convergence to coset 
    alpha_memory::T3
end

br(A, B) = A*B - B*A

@inline function H_of_alpha!(H_of_alpha, H0, l, alpha)
    @inbounds for j in axes(H0,2), i in axes(H0,1)
        H_of_alpha[i,j] = H0[i,j] * exp(alpha * (l[j] - l[i]))
    end
end

function obtain_H_opt(M, p)
    H0, l = p.H0, p.l
    H_of_alpha = similar(H0)
    C = similar(H0)
    alpha_prev = p.alpha_memory[]      

    function cost(a)
        α = a[1]
        H_of_alpha!(H_of_alpha, H0, l, α)
        return -real(tr(H_of_alpha * M))
    end

    alpha0 = [alpha_prev]
    result = optimize(cost, alpha0, NelderMead())
    α_opt = Optim.minimizer(result)[1]   # scalar

    # warm-start for next call
    p.alpha_memory[] = α_opt
    # @show α_opt

    H_of_alpha!(H_of_alpha, H0, l, α_opt)
    return H_of_alpha
end

# function distance_to_target_coset(P_opt, target, p)
#     l = p.l
#     A = adjoint(target) * P_opt
#     a = diag(A)

#     function gprime(α, a, l)
#         y = exp.(α .* l)
#         return real(2 * sum(l .* y .* (y .- a)))
#     end
    
#     αopt = find_zero(α -> gprime(α, diag(A), l), 0.0)
#     y = exp.(αopt .* l)
#     dist = sqrt(sum(abs2, A) - sum(abs2, a) + sum(abs2, a .- y))
    
#     return dist
# end 

function distance_to_target_coset(P, target, params)
    # l = params.l
    # A = adjoint(target) * P
    # a = diag(A)
    # # display(a)

    # # maximize Re tr(e^{-αL} A)
    # cost(α) = -real(sum(exp.(-α .* l) .* a))
    # res = Optim.optimize(cost, [0.0], NelderMead())
    # fmax = -Optim.minimum(res)

    # n = length(a)
    # return sqrt(sum(abs2, A) + n - 2fmax)

    # A = target' * P
    # @show display(A), tr(A)
    # return norm(target - P)
    n = size(P,1)
    A = target' * P
    return sqrt(2n - 2*abs(tr(A))) 

end

# Coupled ODE system, (P_opt, M)
# dP_opt/dt = H_opt(t) * P_opt
# dM/dt = [M, H_opt(t)]
function f!(dX, X, p, t)

    P, M = X.P, X.M
    H_opt = obtain_H_opt(M, p)
    dX.P .= H_opt * P
    dX.M .= br(H_opt, M)
end

function build_M(m, p_basis)
    M = zeros(eltype(p_basis[1]), size(p_basis[1]))
    for (i, m_coeff) in enumerate(m)
        @assert size(p_basis[i]) == size(M)
        M .+= m_coeff * p_basis[i]
    end
    return M
end

function compute_optimal_time(target; tmax = 20.0)

    X = operator(XopRyd([1]), 1)
    Z = operator(ZopRyd([1]), 1)
    gens = [Z, X]
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = lie_basis[2:end]

    params = Params(-im*X, -im*diag(Z), 1e-3, Ref(1.0)) 

    function construct_ODE(m)
        params.alpha_memory[] = 1.0
        M0 = build_M(m, p_basis)
        X0 = ComponentArray(P = id, M = M0)
        ODEProblem(f!, X0, (0.0, tmax), params)
    end

    function objective(m)
        prob = construct_ODE(m)
        sol = solve(prob, saveat=tmax, abstol=1e-8, reltol=1e-8)
        P_T = sol.u[end].P
        dist = distance_to_target_coset(P_T, target, params)
        println("Evaluating at m = $m : dist = $dist")
        
        return dist
    end

    m0 = rand(length(p_basis)) .*2 .-1
    result = optimize(objective, m0, NelderMead())
    # # m_opt = Optim.minimizer(result)
    # # T_opt = Optim.minimum(result)
    println(result)

    # function endpoint_P(m)
    #     prob = construct_ODE(m)
    #     sol = solve(prob, saveat=tmax, abstol=1e-8, reltol=1e-8)
    #     return sol.u[end].P
    # end

    # m1 = [0.7, -0.8, 0.08]
    # m2 = [1.1, -0.8, 0.08]

    # P1 = endpoint_P(m1)
    # P2 = endpoint_P(m2)

    # println("‖P1 - P2‖ = ", norm(P1 - P2))
    # println("‖P1 - I‖  = ", norm(P1 - I))
    # println("‖P2 - I‖  = ", norm(P2 - I))
    # display(P1)
    # display(P2)
    end