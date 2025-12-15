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

# @inline function commutator_diagL!(C, H_of_alpha, Delta)
#     @inbounds @simd for k in eachindex(H_of_alpha)
#         C[k] = Delta[k] * H_of_alpha[k]
#     end
# end

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

    # function cost_grad!(G, a)
    #     α = a[1]
    #     H_of_alpha!(H_of_alpha, H0, Delta, α)
    #     commutator_diagL!(C, H_of_alpha, Delta)
    #     G[1] = -real(tr(C * M))
    # end

    alpha0 = [alpha_prev]
    # result = optimize(cost, cost_grad!, alpha0, LBFGS(); inplace=true)
    result = optimize(cost, alpha0, NelderMead())
    α_opt = Optim.minimizer(result)[1]   # scalar

    # warm-start for next call
    p.alpha_memory[] = α_opt
    # @show α_opt

    H_of_alpha!(H_of_alpha, H0, l, α_opt)
    return H_of_alpha
end


# function get_distance_to_target_coset(P_opt, target, p, alpha_coset_memory)
#     λ = p.l
#     A = adjoint(target) * P_opt
#     # A = P_opt * adjoint(target)

#     function cost(alpha)
#         alpha = alpha[1]
#         return norm(A - spdiagm(0 => exp.(alpha .* λ)))
#     end

#     result = optimize(cost, [alpha_coset_memory[]], NelderMead())
#     alpha_opt = Optim.minimizer(result)[1]
#     if isnan(alpha_opt)
#         alpha_opt = 0.0   # or any safe default
#     end

#     alpha_coset_memory[] = alpha_opt
#     # println(alpha_opt)
#     return cost([alpha_opt])
# end

function distance_to_target_coset(P_opt, target, p)
    l = p.l
    A = adjoint(target) * P_opt
    a = diag(A)

    function gprime(α, a, l)
        y = exp.(α .* l)
        return real(2 * sum(l .* y .* (y .- a)))
    end
    
    αopt = find_zero(α -> gprime(α, diag(A), l), 0.0)
    y = exp.(αopt .* l)
    dist = sqrt(sum(abs2, A) - sum(abs2, a) + sum(abs2, a .- y))
    
    return dist
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

function compute_optimal_time(target; tmax = 20.0, penalty = 10.0)

    # lie_basis = construct_lie_basis_general(gens)
    # @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    # # p_basis = [[lie_basis[1]]; lie_basis[3:end]] # assumption that controls are given by basis element number 2 (after GSO)
    # p_basis = lie_basis
    # L = gens[2]
    # diag_L = diag(L)
    # Delta = diag_L' .- diag_L
    # p = ComponentArray(H0 = -im*gens[1], Delta = Delta, diag_L = diag_L, tol = coset_tolerance, alpha_memory = Ref(1.0)) #is minus causing inconsistencies with lie algebra?
    
    X = operator(XopRyd([1]), 1)
    Z = operator(ZopRyd([1]), 1)
    gens = [Z, X]
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = lie_basis[2:end]

    params = Params(-im*X, -im*diag(Z), 1e-3, Ref(1.0)) 

    function single_sol_run(m; tmax=tmax)
        params.alpha_memory[] = 1.0
        M0 = build_M(m, p_basis)
        X0 = ComponentArray(P = id, M = M0)
        tspan = (0.0, tmax)
        prob = ODEProblem(f!, X0, tspan, params)

        min_dist = Ref(Inf)
        hit_time = Ref(nothing)
        condition(u, t, integrator) = (integrator.iter % 10 == 0)

        function affect!(integrator)
            X = integrator.u
            P, M = X.P, X.M
            params = integrator.p

            dist = distance_to_target_coset(P, target, params)
            if dist < min_dist[]
                min_dist[] = dist
            end 

            if dist < real(params.tol) && hit_time[] === nothing
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

    
    for k in 1:5
        m = rand(length(p_basis))*2 .- 1
        hit_time, min_dist = single_sol_run(m)
        println("run $k: hit_time=$hit_time, min_dist=$min_dist")
    end

    m0 = rand(length(p_basis)) .*2 .-1
    result = optimize(objective, m0, NelderMead())
    m_opt = Optim.minimizer(result)
    T_opt = Optim.minimum(result)
    println(result)
    end