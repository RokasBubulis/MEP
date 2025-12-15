using DifferentialEquations, Optim, FastExpm, ComponentArrays, Roots

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

mutable struct Params{T, T1}
    H0::T
    l::T1
    Hbuf::T                  
    alpha_memory::Base.RefValue{Float64}
end

br(A, B) = A*B - B*A

@inline function H_of_alpha!(H_of_alpha, H0, l, alpha)
    @inbounds for j in axes(H0,2), i in axes(H0,1)
        H_of_alpha[i,j] = H0[i,j] * exp(alpha * (l[j] - l[i]))
    end
end

# to be checked but that could be the way to obtain H_opt without optimisation as L is always diagonal
# function alpha_newton(alpha, H0, M, l)
#     h  = 0.0     # f(α)
#     h_deriv = 0.0     # f'(α)

#     @inbounds for j in axes(H0,2), i in axes(H0,1)
#         delta = l[j] - l[i]
#         z = H0[i,j] * M[j,i] * exp(alpha * delta)
#         h  += real(z)
#         h_deriv += real(delta * z)
#     end

#     return alpha - h / h_deriv
# end

# function obtain_H_opt!(M, p)
#     H0   = p.H0
#     l    = p.l
#     H    = p.Hbuf         

#     α = p.alpha_memory[]
#     α = alpha_newton(α, H0, M, l)
#     p.alpha_memory[] = α

#     H_of_alpha!(H, H0, l, α)   
#     return H                   
# end

# current way to obtain H_opt using optimisation without grad
function obtain_H_opt(M, p)
    H0, l = p.H0, p.l
    H_of_alpha = similar(H0)
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

    H_of_alpha!(H_of_alpha, H0, l, α_opt)
    return H_of_alpha
end

function distance_to_target_coset(P, target, params)
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
    n = size(P,1)
    A = target' * P
    @show display(target), display(P)
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

    params = Params(-im*X, -im*diag(Z), similar(-im*X), Ref(1.0))


    function construct_ODE(m, t)
        params.alpha_memory[] = 1.0
        M0 = build_M(m, p_basis)
        X0 = ComponentArray(P = id, M = M0)
        ODEProblem(f!, X0, (0.0, t), params)
    end

    function objective(x)
        m, t = x[1:end - 1], x[end]
        prob = construct_ODE(m, t)
        sol = solve(prob, saveat=t, abstol=1e-6, reltol=1e-6)
        P_T = sol.u[end].P
        dist = distance_to_target_coset(P_T, target, params)
        println("Evaluating at m = $m , t = $t: dist = $dist")
        
        return dist
    end

    m0 = rand(length(p_basis)) .*2 .-1
    t0 = 10
    tmin = 0.5
    tmax = 30
    # result = optimize(objective, m0, NelderMead())

    n = length(p_basis)
    x0 = [m0; t0]

    lower = [-1 * ones(n); tmin]
    upper = [1 * ones(n); tmax]

    result = optimize(
        objective,
        lower,
        upper,
        x0,
        Fminbox(NelderMead())
    )
    println(result)
    T_opt = Optim.minimum(result)
    P = Optim.minimizer(result)
    println("Minimizer: $P")
    end