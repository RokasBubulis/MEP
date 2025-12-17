using DifferentialEquations, Optim, ComponentArrays, RecursiveArrayTools

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

mutable struct Params
    H0::SparseMatrixCSC{float_type, Int}
    l::SparseVector{float_type}
    n_levels::Int
    n_qubits::Int
    tmin::Float64
    tmax::Float64
    coset_tolerance::Float64  
    dist_coeff::Float64 
    print_intermediate::Bool               
    alpha_memory::Base.RefValue{Float64}
end

# Control of the adjoint system
@inline function H_of_alpha!(H_of_alpha::SparseMatrixCSC{float_type, Int}, 
                            H0::SparseMatrixCSC{float_type, Int}, 
                            l::SparseVector{float_type},
                            alpha::Float64)

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
function obtain_H_opt(M::SparseMatrixCSC{float_type, Int}, params::Params)
    H0, l = params.H0, params.l
    H_of_alpha = similar(H0)
    alpha_prev = params.alpha_memory[]      

    function cost(a)
        α = a[1]
        H_of_alpha!(H_of_alpha, H0, l, α)
        return -real(tr(H_of_alpha * M))
    end

    alpha0 = [alpha_prev]
    result = optimize(cost, alpha0, NelderMead())
    α_opt = Optim.minimizer(result)[1]   # scalar

    # warm-start for next call
    params.alpha_memory[] = α_opt

    H_of_alpha!(H_of_alpha, H0, l, α_opt)
    return H_of_alpha
end

# 1D optimisation to obtain distance to target coset
function distance_to_target_coset(P_opt::SparseMatrixCSC{float_type, Int}, 
                                  target::SparseMatrixCSC{float_type, Int}, params::Params)

    λ = params.l
    A = adjoint(target) * P_opt

    function cost(alpha::Vector{Float64})
        alpha = alpha[1]
        return norm(A - spdiagm(0 => exp.(alpha .* λ)))
    end

    result = optimize(cost, [0.0], NelderMead())
    alpha_opt = Optim.minimizer(result)[1]
    if isnan(alpha_opt)
        alpha_opt = 0.0
    end
    return cost([alpha_opt])
end

# Coupled ODE system, (P_opt, M)
# dP_opt/dt = H_opt(t) * P_opt
# dM/dt = [H_opt(t), M]
# function f!(dX, X, p, t)

#     P, M = X.P, X.M
#     H_opt = obtain_H_opt(M, p)
#     dX.P .= H_opt * P
#     dX.M .= br(H_opt, M)
# end
function f!(dX, X, p, t)
    P = X.x[1]
    M = X.x[2]

    H_opt = obtain_H_opt(M, p)

    dP = H_opt * P
    dM = br(H_opt, M)

    # Write into existing storage (do NOT reassign tuple entries)
    dX.x[1] .= dP
    dX.x[2] .= dM
end

# Build costate M from the basis of the orthogonal complement of control subalgebra
function build_M(m::Vector{Float64}, p_basis::Vector{SparseMatrixCSC{float_type, Int}})

    M = spzeros(float_type, size(p_basis[1])...)

    for (i, m_coeff) in enumerate(m)
        M .+= m_coeff * p_basis[i]
    end

    return M
end

###############
# Main function
function compute_optimal_time(gens::Vector{SparseMatrixCSC{float_type, Int}}, 
                              target::SparseMatrixCSC{float_type, Int}, params::Params)
    # Note: this function assumes 1 control given as the first generator and 1 drift as the second generator!
    
    # Check if target is implementable and construct the basis of orthogonal complement of control subalgebra
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = lie_basis[2:end] 
    dim = params.n_levels ^ params.n_qubits
    P0 = spdiagm(0 => ones(float_type, dim))

    # Construct a single shoot: obtain geodesic for a given time and initial costate (=momentum)
    function construct_ODE(m::Vector{Float64}, t::Float64)
        params.alpha_memory[] = 1.0
        M0 = build_M(m, p_basis)
        X0 = ArrayPartition(copy(P0), M0)
        ODEProblem(f!, X0, (0.0, t), params)
    end

    # Objective function that drives down time and final distance to target coset
    function objective(x::Vector{Float64})
        m, t = x[1:end - 1], x[end]
        prob = construct_ODE(m, t)
        sol = solve(prob, saveat=t, abstol=1e-6, reltol=1e-6)
        if sol.retcode != SciMLBase.ReturnCode.Success
            return 1e20
        end
        P_T = sol.u[end].x[1]
        dist = distance_to_target_coset(P_T, target, params)
        if params.print_intermediate
            println("Evaluating at m = $m, t = $t : dist = $dist")
        end
                
        return params.dist_coeff * dist^2 + t
    end
    
    # Ensure valid parameters by penalising large values
    function objective_penalized(x::Vector{Float64})
        m, t = x[1:end-1], x[end]
        penalty = 0.0
        penalty += sum(max.(0, abs.(m) .- 1).^2) * 1e6
        penalty += max(0, params.tmin - t)^2 * 1e6
        penalty += max(0, t - params.tmax)^2 * 1e6

        return objective(x) + penalty
    end

    # Initial guess
    m0 = rand(length(p_basis)) .*2 .-1
    t0 = 2
    x0 = [m0; t0]

    # Optimisation
    result = optimize(
        objective_penalized,
        x0,
        NelderMead()
    )
    
    # Results
    println(result)
    x_final = Optim.minimizer(result)
    final_objective = Optim.minimum(result)
    final_dist = (final_objective - x_final[end])^(1/2) / params.dist_coeff
    if abs(final_dist) > params.coset_tolerance
        error("Target coset has not been reached at time $(x_final[end]/pi) π")
    else
        println(("Target coset has been reached at time $(x_final[end]/pi) π"))
    end
    println("Minimizer: $x_final")

end