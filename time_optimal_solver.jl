using DifferentialEquations, Optim, RecursiveArrayTools, Dates

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
    turning_point_factor::Float64
    coset_hard_tol::Float64
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
function f!(dX, X, p, t)
    P = X.x[1]
    M = X.x[2]

    H_opt = obtain_H_opt(M, p)
    dP = H_opt * P
    dM = br(H_opt, M)

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
    println("dim(lie basis) = $(length(lie_basis))")
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = lie_basis[2:end] 
    dim = params.n_levels ^ params.n_qubits
    P0 = spdiagm(0 => ones(float_type, dim))

    # Construct a single shoot: obtain geodesic for a given time and initial costate (=momentum)
    function construct_ODE(m::Vector{Float64}, t::Float64)
        params.alpha_memory[] = 1.0
        M0 = build_M(m, p_basis)
        @assert M0 !== nothing "build_M returned nothing"
        X0 = ArrayPartition(copy(P0), M0)
        ODEProblem(f!, X0, (0.0, t), params)
    end

    function distance_at_time(m::Vector{Float64}, t::Float64)
        m_normalised = m / max(norm(m), 1e-12)
        prob = construct_ODE(m_normalised, t)
        sol = solve(prob, saveat=t, abstol=1e-4, reltol=1e-4)
        sol.retcode != SciMLBase.ReturnCode.Success && return 1e20

        P_T = sol.u[end].x[1]
        return distance_to_target_coset(P_T, target, params)
    end

    function optimize_m_for_time(t, m0)
        obj(m) = distance_at_time(m, t)
        res = optimize(obj, m0, NelderMead(), Optim.Options(iterations = 30))
        m_best = Optim.minimizer(res)
        dist_best = distance_at_time(m_best, t)
        return dist_best, m_best, res
    end

    # find the interval in which distance to target coset starts again increasing
    Δ = params.turning_point_factor
    function find_local_optimum(t1, m_best)
        t_left = 0.0
        t_right = 0.0

        # First point
        d1, m1, _ = optimize_m_for_time(t1, m_best)
        println("Checking at t = $t1: dist = $d1")
        # Second point
        t2 = t1 * Δ
        d2, m2, _ = optimize_m_for_time(t2, m1)
        println("Checking at t = $t2: dist = $d2")
        turning_point = false

        while t2 ≤ params.tmax
            t3 = t2 * Δ
            d3, m3, _ = optimize_m_for_time(t3, m2)

            if params.print_intermediate
                println("Checking at t = $t3: dist = $d3")
            end
            
            # Check if a local minimum of distance to target coset
            if d2 < d1 && d2 < d3
                t_left = t1
                t_right = t3
                m_best = m2
                println("Turning point found, searching for local minimum")
                turning_point = true
                break
            end

            t1, d1 = t2, d2
            t2, d2, m2 = t3, d3, m3
        end

        turning_point || error("Local minimum not found up to tmax = $(params.tmax)")

        # Given time interval when distance starts increasing, find minimal time
        φ(t) = begin
            d, m, _ = optimize_m_for_time(t, m_best)
            d
        end
        res = optimize(φ, t_left, t_right, Brent(); iterations = 30, rel_tol = 1e-5, abs_tol = 1e-5)

        t_star = Optim.minimizer(res)
        d_star = Optim.minimum(res)
        return t_star, d_star, t2, m2
    end

    # First local minimum
    m_best = rand(length(p_basis)) .*2 .-1
    t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(params.tmin, m_best)

    while true

        if d_loc_min < params.coset_hard_tol
            println("Local optimum converged to target coset within tolerance")
            break
        else
            println("Local optimum at t = $t_loc_min did not converge to target coset within tolerance, continuing")
            t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(last_t * Δ, m_best)
        end

        if last_t >= params.tmax
            error("No local optimum converged to target coset within tolerance up to tmax = $(params.tmax)")
        end
    end

    println("Optimal time: $(t_loc_min/pi) π, distance to target coset: $d_loc_min")

end