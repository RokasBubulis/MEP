using DifferentialEquations, Optim, Roots, LinearAlgebra

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")
include("plot_geodesics.jl")

mutable struct Params{T}
    H0::SparseMatrixCSC{T, Int}
    l::Vector{T}
    p_basis::Vector{SparseMatrixCSC{T, Int}}
    V_Ryd::Vector{T}
    tmin::Float64
    tmax::Float64
    turning_point_factor::Float64
    coset_hard_tol::Float64
    print_intermediate::Bool
    previous_alpha::Float64
    previous_gamma::Float64
    dim::Int

    H_temp::SparseMatrixCSC{T, Int}  

    tmp1::Matrix{T}                  
    tmp2::Matrix{T}              
end

function H_of_alpha!(H_of_alpha::SparseMatrixCSC, H0::SparseMatrixCSC, l::Vector, alpha::Float64)
    @inbounds for col in 1:size(H0,2)
        for ptr in H0.colptr[col]:(H0.colptr[col+1]-1)
            row = H0.rowval[ptr]
            H_of_alpha.nzval[ptr] = H0.nzval[ptr] * exp(alpha * (l[col] - l[row]))
        end
    end
end

function H_opt!(M::AbstractMatrix, params::Params)
    H0, l, α0 = params.H0, params.l, params.previous_alpha

    function f(x)
        α = x[1]
        func = 0.0

        @inbounds for col in 1:size(H0,2)
            for ptr in H0.colptr[col]:(H0.colptr[col+1]-1)
                row = H0.rowval[ptr]
                if row < col
                    Δ = l[col] - l[row]
                    e = exp(α * Δ)
                    z = H0.nzval[ptr] * M[col,row]
                    func += real(e * z)
                end
            end
        end
        return -func
    end

    function g!(G, x)
        α = x[1]
        grad = 0.0

        @inbounds for col in 1:size(H0,2)
            for ptr in H0.colptr[col]:(H0.colptr[col+1]-1)
                row = H0.rowval[ptr]
                if row < col
                    Δ = l[col] - l[row]
                    e = exp(α * Δ)
                    z = H0.nzval[ptr] * M[col,row]
                    grad += real(Δ * e * z)
                end
            end
        end
        G[1] = -grad
        return nothing
    end

    x0 = [α0]
    res = optimize(f, g!, [-pi/2], [pi/2], x0, Fminbox(BFGS()))
    α_opt = Optim.minimizer(res)[1]
    params.previous_alpha = α_opt
    H_of_alpha!(params.H_temp, H0, l, α_opt)
end 

# #1D optimisation to obtain distance to target coset
# function distance_to_target_coset(P_opt::AbstractMatrix, 
#                                   target::SparseMatrixCSC{float_type, Int}, params::Params)

#     λ = params.l
#     #A = adjoint(target) * P_opt
#     A = P_opt * adjoint(target)

#     function cost(alpha::Vector{Float64})
#         alpha = alpha[1]
#         return norm(A - spdiagm(0 => exp.(alpha .* λ)))
#     end

#     result = optimize(cost, [0.0], NelderMead())
#     alpha_opt = Optim.minimizer(result)[1]
#     if isnan(alpha_opt)
#         alpha_opt = 0.0
#     end
#     return cost([alpha_opt])
# end

function distance_to_target_coset(P_opt::AbstractMatrix,
                                  target::SparseMatrixCSC{float_type, Int},
                                  params::Params)

    l = params.l
    A = P_opt * adjoint(target)
    normA = sum(abs2, A)          
    a  = diag(A)                     

    @inline function distance(gamma::Float64)
        d = exp.(gamma .* l)     
        return sqrt(normA + sum(abs2, d) - 2 * real(dot(a, d)))
    end

    result = optimize(distance, -pi, pi)
    gamma_opt = Optim.minimizer(result)
    return distance(gamma_opt)
end

function plot_results(sol, target, params, name)
    dim = params.dim
    dim2 = dim^2
    p_vals_rot = Vector{Float64}(undef, length(sol.u))
    p_vals_lab = Vector{Float64}(undef, length(sol.u))
    coset_vals_rot = Vector{Float64}(undef, length(sol.u))
    coset_vals_lab = Vector{Float64}(undef, length(sol.u))
    tv = vec(target)

    label = nothing
    Plab_best = nothing
    for k in eachindex(sol.u)
        u = sol.u[k]
        t = sol.t[k]

        @views Pk = reshape(view(u, 1:dim2), dim, dim)

        U0 = spdiagm(0 => exp.(-im .* params.V_Ryd .* t))
        P_lab = U0 * Pk
        p_vals_rot[k] = abs(dot(vec(Pk), tv)) / (norm(Pk) * norm(tv))
        p_vals_lab[k] = abs(dot(vec(P_lab), tv)) / (norm(P_lab) * norm(tv))
        coset_vals_rot[k] = distance_to_target_coset(Pk, target, params)
        coset_vals_lab[k] = distance_to_target_coset(P_lab, target, params)
        if coset_vals_lab[k] < 0.1
            label = k
            Plab_best = P_lab
        end 
        if abs( 1- p_vals_rot[k]) < 0.1

            A1 = P_lab * target'
            A2 = target' * P_lab

            off1 = norm(A1 - Diagonal(diag(A1)))
            off2 = norm(A2 - Diagonal(diag(A2)))

            println(off1, off2)
        end
    end
    println("Closest time: $(sol.t[label]), distance: $(coset_vals_lab[label])")
    display(Plab_best)

    plt = plot(
        plot(sol.t, [p_vals_rot p_vals_lab];
            title = "Overlap with target",
            xlabel = "t",
            ylabel = "|⟨P(t), target⟩|",
            label = ["Rot frame" "Lab frame"]
        ),
        plot(sol.t, [coset_vals_rot coset_vals_lab];
            title = "Distance to target coset",
            xlabel = "t",
            ylabel = "Coset distance",
            label = ["Rot frame" "Lab frame"]
        ),
        layout = (2, 1),   # 2 rows, 1 column
        size = (800, 800)
    )
    savefig(plt, "$(name)_visualisation.png")

    return plt
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
    dim = params.dim
    dim2 = dim^2

    @views P  = reshape(view(X, 1:dim2), dim, dim)
    @views M  = reshape(view(X, dim2+1:2*dim2), dim, dim)
    @views dP = reshape(view(dX, 1:dim2), dim, dim)
    @views dM = reshape(view(dX, dim2+1:2*dim2), dim, dim)

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
function compute_optimal_time(target::SparseMatrixCSC{float_type, Int}, params::Params{T}, name::String) where {T}
    # Note: this function assumes 1 control given as the first generator and 1 drift as the second generator!
    p_basis = params.p_basis
    dim = params.dim
    dim2 = dim^2
    P0 = spdiagm(0 => ones(float_type, dim))
    X0 = Vector{T}(undef, 2*dim2)
    X0[1:dim2] .= vec(P0)

    # Construct a single shoot: obtain geodesic for a given time and initial costate (=momentum)
    function construct_ODE(m::Vector{Float64}, t::Float64, p_basis, params::Params{T}) where {T}
        M0_sparse = build_M(m, p_basis)
        M0 = Matrix{T}(M0_sparse)                 
        X0[dim2+1:end] .= vec(M0)
        ODEProblem(f!, copy(X0), (0.0, t), params)
    end

    function distance_at_time(m::Vector{Float64}, t::Float64)
        prob = construct_ODE(m, t, p_basis, params)
        sol = solve(prob, Tsit5();
        abstol=1e-5, reltol=1e-5,
        save_everystep=false, save_start=false
    )
        sol.retcode != SciMLBase.ReturnCode.Success && return 1e20

        @views P_T = reshape(view(sol.u[end], 1:dim2), dim, dim)
        # convert to lab frame
        U0 = Diagonal(exp.(-im .* params.V_Ryd .* t))
        P_lab = U0 * P_T
        return distance_to_target_coset(P_lab, target, params)
    end

    function optimize_m_for_time(t, m0)
        obj(m) = distance_at_time(m, t)
        res = optimize(obj, m0, NelderMead(), Optim.Options(iterations = 1000))
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
            res = optimize(φ, t_left, t_right, Brent(); iterations = 500, rel_tol = 1e-8, abs_tol = 1e-8)

            t_star = Optim.minimizer(res)
            d_star = Optim.minimum(res)
            return t_star, d_star, t2, m2
        else
            return t2, d2, t2, m2
        end
    end

    dself = distance_to_target_coset(Matrix(target), target, params)
    U = Matrix(target)
    dim = size(U,1)
    @assert norm(U' * U - I(dim), Inf) ≤ 1e-10 "Target not unitary"
    @assert dself < params.coset_hard_tol "Target not in target coset within tolerance"

    # First local minimum
    m_best = rand(length(p_basis)) .*2 .-1

    # M = build_M(m_best, p_basis)
    # H_opt!(M, params)
    # function Φ(α, H0, l, M)
    #     n = size(H0, 1)
    #     s = 0.0
    #     @inbounds for j in 1:n, i in 1:n
    #         Δ = l[j] - l[i]
    #         s += real(exp(α * Δ) * H0[i,j] * M[j,i])
    #     end
    #     return s
    # end
    # alphas = range(-π/2, π/2; length=200)
    # vals = [Φ(a, params.H0, params.l, M) for a in alphas]
    # plt = plot(alphas, vals, xlabel="α", ylabel="Re tr(H(α) M)")
    # vline!([params.previous_alpha])
    # savefig(plt, "$(name)_optimal_alpha.png")

    
    t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(params.tmin, m_best)

    while true

        if d_loc_min < params.coset_hard_tol

            if print_intermediate
                println("Local optimum converged to target coset within tolerance")
            end
            break
        else
            print_intermediate && println("Local optimum $d_loc_min at t = $t_loc_min did not converge to target coset within tolerance of $(params.coset_hard_tol), continuing")
            println("t=$t_loc_min, d=$d_loc_min")
            prob = construct_ODE(m_best, t_loc_min, p_basis, params)
            sol = solve(prob, Tsit5(), save_everystep=false, save_start=false, abstol=1e-8, reltol=1e-8)
            P = Vector{Matrix{float_type}}(undef, length(sol.u))
            for (k, u) in pairs(sol.u)
                P[k] = reshape(view(u, 1:dim2), dim, dim)
            end
            println("P_rot:")
            display(P[end])
            U0 = spdiagm(0 => exp.(-im .* params.V_Ryd .* sol.t[end]))
            println("P_lab:")
            display(U0 * P[end])
            t_loc_min, d_loc_min, last_t, m_best = find_local_optimum(last_t * Δ, m_best)
        end

        if last_t >= params.tmax
            prob = construct_ODE(m_best, t_loc_min, p_basis, params)
            sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=1000), abstol=1e-9, reltol=1e-9)
            P = Vector{Matrix{float_type}}(undef, length(sol.u))
            for (k, u) in pairs(sol.u)
                @views P[k] = reshape(view(u, 1:dim2), dim, dim)
            end
            display(plot_results(sol, target, params, name))
            error("No local optimum converged to target coset within tolerance up to tmax = $(params.tmax)")
        end
    end

    print_intermediate && println("Optimal time: $(t_loc_min/pi) π, distance to target coset: $d_loc_min")
    prob = construct_ODE(m_best, t_loc_min, p_basis, params)
    sol = solve(prob, Tsit5(), saveat=range(0.0, t_loc_min, length=50), abstol=1e-4, reltol=1e-4)
    display(plot_results(sol, target, params, name))
    #display(plot_optimal_geodesic(sol, gens, target, params, t_loc_min))
    P = Vector{Matrix{float_type}}(undef, length(sol.u))
    for (k, u) in pairs(sol.u)
        @views P[k] = reshape(view(u, 1:dim2), dim, dim)
    end
    println("P_rot:")
    display(P[end])
    return P

end