using DifferentialEquations, Optim, Roots

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")
include("plot_geodesics.jl")

mutable struct Params{T}
    H0::SparseMatrixCSC{T, Int}
    l::SparseVector{T}
    tmin::Float64
    tmax::Float64
    turning_point_factor::Float64
    coset_hard_tol::Float64
    print_intermediate::Bool
    previous_alpha::Float64
    previous_gamma::Float64
    dim::Int
    dim2::Int

    H_temp::SparseMatrixCSC{T, Int}  

    tmp1::Matrix{T}                  
    tmp2::Matrix{T}              
end

function make_params(H0::SparseMatrixCSC{T,Int}, l::SparseVector{T};
                     tmin::Float64, tmax::Float64,
                     turning_point_factor::Float64,
                     coset_hard_tol::Float64,
                     print_intermediate::Bool=false,
                     previous_alpha::Float64=0.0,
                     previous_gamma::Float64=0.0) where {T}

    dim = size(H0, 1)
    @assert size(H0,2) == dim "H0 must be square"
    dim2 = dim*dim

    H_temp = copy(H0)        
    tmp1   = Matrix{T}(undef, dim, dim)
    tmp2   = Matrix{T}(undef, dim, dim)

    return Params{T}(H0, l, tmin, tmax,
                     turning_point_factor, coset_hard_tol,
                     print_intermediate, previous_alpha, previous_gamma,
                     dim, dim2,
                     H_temp, tmp1, tmp2)
end


function H_of_alpha!(H_of_alpha::SparseMatrixCSC, H0::SparseMatrixCSC, l::SparseVector, alpha::Float64)
    @inbounds for col in 1:size(H0,2)
        for ptr in H0.colptr[col]:(H0.colptr[col+1]-1)
            row = H0.rowval[ptr]
            H_of_alpha.nzval[ptr] = H0.nzval[ptr] * exp(alpha * (l[col] - l[row]))
        end
    end
end

# function H_opt!(M::AbstractMatrix, params::Params)
#     H0, l, α0 = params.H0, params.l, params.previous_alpha
    
#     function pmp_derivative(α)
#         n = size(H0, 1)
#         s = 0
#         for j in 2:n, i in 1:j-1
#             Δ = l[j] - l[i]
#             s += real(Δ*exp(α*Δ)*H0[i,j]*M[j,i])
#         end
#         return s
#     end

#     α_opt = fzero(pmp_derivative, α0)
#     n = size(H0, 1)
#     s = 0
#     for j in 2:n, i in 1:j-1
#         Δ = l[j] - l[i]
#         s += real(Δ^2*exp(Δ * α_opt)*H0[i,j]*M[j,i])
#     end
#     @assert s < 0 "2nd derivative not negative: $s"
#     params.previous_alpha = α_opt
#     H_of_alpha!(params.H_temp, H0, l, α_opt)
#     return
# end

function pmp_d1_d2(α, H0, l, M)
    d1 = 0.0
    d2 = 0.0
    n = size(H0, 1)
    for j in 2:n, i in 1:j-1
        Δ = l[j] - l[i]
        e = exp(Δ*α)
        z = H0[i,j]*M[j,i]
        d1 += real(Δ*e*z)
        d2 += real(Δ^2*e*z)
    end
    return d1, d2
end


function H_opt!(M::AbstractMatrix, params::Params)
    H0 = params.H0
    l  = params.l
    α  = params.previous_alpha

    # 2–3 Newton steps
    @inbounds for _ in 1:3
        g  = 0.0
        gp = 0.0

        for col in 1:size(H0,2)
            for ptr in H0.colptr[col]:(H0.colptr[col+1]-1)
                row = H0.rowval[ptr]

                if row < col
                    Δ = l[col] - l[row]
                    e = exp(α * Δ)
                    hij = H0.nzval[ptr]
                    mji = M[col, row]     

                    term = Δ * e * hij * mji
                    g  += real(term)
                    gp += real((Δ*Δ) * e * hij * mji)
                end
            end
        end

        # guard against division by tiny derivative
        abs(gp) < 1e-14 && break
        α -= g / gp
    end

    params.previous_alpha = α

    H_of_alpha!(params.H_temp, params.H0, params.l, α)
    return nothing
end

# function distance_to_target_coset(P_opt::AbstractMatrix, 
#                                   target::SparseMatrixCSC{float_type, Int}, params::Params)
#     A_adjoint = adjoint(P_opt) * target  # A = adjoint(U_T) * P_T
#     a_adj_diag = diag(A_adjoint)
#     l = params.l
#     gamma_0 = params.previous_gamma

#     function derivative(gamma) # minimises distance^2
#         s = 0.0
#         for i in eachindex(a_adj_diag)
#             s += real(a_adj_diag[i]*l[i]*exp(gamma * l[i]))
#         end
        
#         return s
#     end
    
#     gamma_opt = fzero(derivative, gamma_0)
#     params.previous_gamma = gamma_opt
#     B = spdiagm(exp.(gamma_opt .*l))

#     return norm(adjoint(A_adjoint) - B)
# end


#1D optimisation to obtain distance to target coset
function distance_to_target_coset(P_opt::AbstractMatrix, 
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

function br!(dM::StridedMatrix, H::SparseMatrixCSC, M::StridedMatrix, params)
    # tmp1 = H*M   (dense)
    mul!(params.tmp1, H, M)

    # tmp2 = M*H   (dense)
    mul!(params.tmp2, M, H)

    @. dM = params.tmp1 - params.tmp2
    return nothing
end

function f!(dX, X, params, t)
    dim, dim2 = params.dim, params.dim2

    @views P  = reshape(view(X, 1:dim2), dim, dim)
    @views M  = reshape(view(X, dim2+1:2dim2), dim, dim)
    @views dP = reshape(view(dX, 1:dim2), dim, dim)
    @views dM = reshape(view(dX, dim2+1:2dim2), dim, dim)

    H_opt!(M, params) 

    mul!(dP, params.H_temp, P)
    br!(dM, params.H_temp, M, params)

    return nothing
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
                              target::SparseMatrixCSC{float_type, Int}, params::Params, name::String)
    # Note: this function assumes 1 control given as the first generator and 1 drift as the second generator!
    
    # Check if target is implementable and construct the basis of orthogonal complement of control subalgebra
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"
    p_basis = lie_basis[2:end] 
    dim = params.dim
    dim2 = params.dim2
    P0 = spdiagm(0 => ones(float_type, dim))

    # Construct a single shoot: obtain geodesic for a given time and initial costate (=momentum)
    function construct_ODE(m::Vector{Float64}, t::Float64, p_basis, params::Params{T}) where {T}
        m_normalised = m / max(norm(m), 1e-12)
        M0_sparse = build_M(m_normalised, p_basis)
        M0 = Matrix{T}(M0_sparse)                 
        P0 = Matrix{T}(I, params.dim, params.dim)

        X0 = Vector{T}(undef, 2*params.dim2)
        X0[1:params.dim2] .= vec(P0)
        X0[params.dim2+1:end] .= vec(M0)
        ODEProblem(f!, X0, (0.0, t), params)
    end

    function distance_at_time(m::Vector{Float64}, t::Float64)
        prob = construct_ODE(m, t, p_basis, params)
        sol = solve(prob, Tsit5(), saveat=t, abstol=1e-4, reltol=1e-4)
        sol.retcode != SciMLBase.ReturnCode.Success && return 1e20

        @views P_T = reshape(view(sol.u[end], 1:dim2), dim, dim)
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
        print_intermediate && println("Checking at t = $t1: dist = $d1")
        # Second point
        t2 = t1 * Δ
        d2, m2, _ = optimize_m_for_time(t2, m1)
        print_intermediate && println("Checking at t = $t2: dist = $d2")
        turning_point = false

        while t2 ≤ params.tmax
            t3 = t2 * Δ
            d3, m3, _ = optimize_m_for_time(t3, m2)
            print_intermediate && println("Checking at t = $t3: dist = $d3")
            
            # Check if a local minimum of distance to target coset
            if d2 < d1 && d2 < d3
                t_left = t1
                t_right = t3
                m_best = m2
                print_intermediate && println("Turning point found, searching for local minimum")
                turning_point = true
                break
            end

            t1, d1 = t2, d2
            t2, d2, m2 = t3, d3, m3
        end

        #turning_point  || error("Local minimum not found up to tmax = $(params.tmax)")
        if turning_point
            # Given time interval when distance starts increasing, find minimal time
            φ(t) = begin
                d, m, _ = optimize_m_for_time(t, m_best)
                d
            end
            res = optimize(φ, t_left, t_right, Brent(); iterations = 50, rel_tol = 1e-5, abs_tol = 1e-5)

            t_star = Optim.minimizer(res)
            d_star = Optim.minimum(res)
            return t_star, d_star, t2, m2
        else
            return t2, d2, t2, m2
        end
    end

    # First local minimum
    m_best = rand(length(p_basis)) .*2 .-1

    # M = build_M(m_best, p_basis)
    # l = params.l
    # function Φ(α, H0, l, M)
    #     n = size(H0, 1)
    #     s = 0.0
    #     @inbounds for j in 1:n, i in 1:n
    #         Δ = l[j] - l[i]
    #         s += real(Δ* exp(α * Δ) * H0[i,j] * M[j,i])
    #     end
    #     return s
    # end
    # println(m_best)
    # alphas = range(-π/2, π/2; length=200)
    # vals = [Φ(a, params.H0, l, M) for a in alphas]

    # display(plot(alphas, vals, xlabel="α", ylabel="Re tr(H(α) M)"))
    
    t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(params.tmin, m_best)

    while true

        if d_loc_min < params.coset_hard_tol

            if print_intermediate
                println("Local optimum converged to target coset within tolerance")
                prob = construct_ODE(m_best, t_loc_min, p_basis, params)
                sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=50), abstol=1e-4, reltol=1e-4)
                display(plot_optimal_geodesic(sol, gens, target, params, t_loc_min))
            end
            break
        else
            print_intermediate && println("Local optimum $d_loc_min at t = $t_loc_min did not converge to target coset within tolerance of $(params.coset_hard_tol), continuing")
            # prob = construct_ODE(m_best, t_loc_min, p_basis, params)
            # sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=100), abstol=1e-4, reltol=1e-4)
            println("t=$t_loc_min, d=$d_loc_min")
            prob = construct_ODE(m_best, t_loc_min, p_basis, params)
            sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length = 50), abstol=1e-4, reltol=1e-4)
            P = Vector{Matrix{float_type}}(undef, length(sol.u))
            # M = Vector{Matrix{float_type}}(undef, length(sol.u))
            for (k, u) in pairs(sol.u)
                P[k] = reshape(view(u, 1:dim2), dim, dim)
                # M[k] = reshape(view(u, dim2+1:2*dim2), dim, dim)
            end
            display(P[end])
            t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(last_t * Δ, m_best)
        end

        if last_t >= params.tmax
            prob = construct_ODE(m_best, t_loc_min, p_basis, params)
            sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=100), abstol=1e-4, reltol=1e-4)
            P = Vector{Matrix{Float64}}(undef, length(sol.u))
            M = Vector{Matrix{Float64}}(undef, length(sol.u))
            for (k, u) in pairs(sol.u)
                @views P[k] = reshape(view(u, 1:dim2), dim, dim)
                @views M[k] = reshape(view(u, dim2:2*dim2), dim, dim)
            end
            display(plot_p_inner_target(sol, target, params, name))
            println("No local optimum converged to target coset within tolerance up to tmax = $(params.tmax)")
        end
    end

    print_intermediate && println("Optimal time: $(t_loc_min/pi) π, distance to target coset: $d_loc_min")
    prob = construct_ODE(m_best, t_loc_min, p_basis, params)
    sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=100), abstol=1e-4, reltol=1e-4)
    display(plot_p_inner_target(sol, target, params, name))

end