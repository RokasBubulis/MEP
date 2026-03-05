using DifferentialEquations, Optim, Roots, LinearAlgebra, Plots, CMAEvolutionStrategy

include("generators.jl")
include("lie_algebra.jl")
include("group_implementability.jl")

T = float_type

mutable struct  Params{T}
    drift:: SparseMatrixCSC{T, Int} # H0
    diagonal_control:: SparseMatrixCSC{T, Int} # sum_i Z_i
    diagonal_control_vec:: Vector{T}
    tmin::Float64 # PMP prop t0
    tmax::Float64 # PMP prop tmax
    time_coeff::Float64 # 
    coset_tol::Float64
    abs_tol::Float64
    rel_tol::Float64
    beta::Float64
    dim::Int
    H_alpha_tmp::Matrix{T}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
end

# TODO: This calculation seems to be wrong. Use Campbell identity: https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
# "Control adjusted drift. Ad_k"
# function H_α!(H_α::AbstractMatrix, drift:: AbstractMatrix, diag_control::Vector, α::Float64)
#     n = size(drift, 1)
#     @inbounds for i in 1:n, j in 1:n
#         H_α[i,j] = drift[i,j] * exp(α * (diag_control[j] - diag_control[i]))
#     end
#     return nothing
# end

# function H_optimal!(M::AbstractMatrix, params::Params)
#     drift, l = params.drift, params.diag_control
#     n = size(drift, 1)

#     function f(x, drift, M) # Commutation assumption does not apply
#         α = x[1]
#         func = 0.0
#         @inbounds for i in 1:n, j in 1:n
#             Δ = l[j] - l[i]
#             e = exp(α * Δ)
#             z = drift[i,j] * M[j,i]
#             func += real(e*z)
#         end

#         return -func
#     end

#     # Check derivative numerically! g(x) ≃ (f(x+dt) - f(x-dt)) / (2 dt)
#     function g!(G, x)
#         α = x[1]
#         grad = 0.0
#         @inbounds for i in 1:n, j in 1:n
#             Δ = l[j] - l[i]
#             e = exp(α * Δ)
#             z = drift[i,j] * M[j,i]
#             grad += real(Δ * e * z)
#         end
#         G[1] = -grad
#         return nothing
#     end 
#     # anonymous functions: (x)->(f(x, args)), or optimizer args
#     # Include second derivative, use Newton()
#     # Use linesearch=LineSearches.BackTracking()
#     res = optimize(f, g!, [-Float64(π/2)], [Float64(π/2)], [0.0], Fminbox(BFGS()))
#     α_opt = Optim.minimizer(res)[1]
#     H_α!(params.H_alpha, drift, l, α_opt)
#     return nothing
# end

# distance_to_target_coset_if_diagonal
function distance_to_target_coset(P_opt::AbstractMatrix, # P_M0
                                  target::SparseMatrixCSC{float_type, Int}, # U_T
                                  params::Params)

    A = P_opt * adjoint(target)
    normA = sum(abs2, A)          
    a  = diag(A)                     

    @inline function distance(gamma::Float64)
        d = exp.(gamma .* params.diagonal_control_vec)     
        return sqrt(normA + sum(abs2, d) - 2 * real(dot(a, d)))
    end

    # do it inplace. Also avoid functions inside functions 
    # @inline function distance!(d, gamma::Float64)
    #     d[:] .= exp.(gamma .* l)     
    #     return sqrt(normA + sum(abs2, d) - 2 * real(dot(a, d)))
    # end

    result = optimize(distance, -pi, pi)
    gamma_opt = Optim.minimizer(result)
    return distance(gamma_opt)
end

# These are elements of the algebra, use structure constant (future scalability thing)
# struct LieOperator
#   hilbert::HilbertOperator (~ matrix)
#   lie_vec::LieElement (~ vec)
#   Lie_adjoint::LieOp (~ matrix)
# end
# Also types
function br!(dM, H, M, params)
    mul!(params.tmp1, H, M)
    mul!(params.tmp2, M, H)
    @. dM[:] = params.tmp1 - params.tmp2
    return nothing 
end

"main PMP propagator"
function f!(dX, X, params, t)
    dim = params.dim
    dim2 = dim ^ 2
    # Use normal matrices, or LieOperator
    @views P = reshape(view(X, 1:dim2), dim, dim)
    @views M = reshape(view(X, dim2 + 1: 2*dim2), dim, dim)
    @views dP = reshape(view(dX, 1:dim2), dim, dim)
    @views dM = reshape(view(dX, dim2 + 1: 2*dim2), dim, dim)

    H_optimal!(M, params)
    mul!(dP, params.H_alpha, P)
    br!(dM, params.H_alpha, M, params)

    return nothing
end

function build_M!(M::AbstractMatrix{T}, m::AbstractVector{Float64}, p_basis::Vector{SparseMatrixCSC{T, Int}}) where {T<:Complex}
    M[:] .= 0
    # fill!(M, zero(T))
    for (i, m_coeff) in enumerate(m)
        M[:] .+= m_coeff * p_basis[i]
    end
    return nothing
end 

#########################################

function main_function()
    # Settings
    tmin = 0.1
    tmax = 5 * pi
    time_coeff = 1e-1
    coset_tol = 1e-6
    abs_tol = 1e-6
    rel_tol = 1e-6
    beta = 5.0

    n_saves = 1000

    # One qubit case
    # n_qubits = 1
    # n_levels = 2
    # drift = operator(Xop([1]), n_qubits)
    # control = operator(Zop([1]), n_qubits)
    # target = -im* operator(Yop([1]), n_qubits)

    # # Two qubit case
    # n_qubits = 2
    # n_levels = 2
    # drift = operator(Xop([1, 2]), n_qubits)
    # control = operator(Zop([1]), n_qubits) + operator(Zop([2]), n_qubits)
    # target = construct_CZ_target(n_qubits, n_levels)
    # positions = [0 0; 0 1]

    function construct_YQ_target(n_qubits::Int)
        A = spzeros(float_type, 3^n_qubits, 3^n_qubits)
        for i in 1:n_qubits
            Qnot = sparse(Matrix{float_type}(I, 3^n_qubits, 3^n_qubits))
            for j in 1:n_qubits
                if j != i 
                    Qnot *= operator(QopRyd([j]), n_qubits)
                end
            end
            A += operator(ZopRyd([i]), n_qubits) * Qnot
        end
        return A
    end

    # Two qutrit case
    n_qubits = 2
    n_levels = 3
    control, drift = construct_Ryd_generators(n_qubits)
    target = sparse(exp(-im*Matrix(construct_YQ_target(n_qubits))))
    # target = sparse(exp(-im*Matrix(drift)))
    display(target)
    # target = construct_CZ_target(n_qubits, n_levels)  # only for two qutrits
    # target = operator(YopRyd([1]), n_qubits) + operator(YopRyd([2]), n_qubits)

    ##############################
    gens = [control, drift]
    dim = size(gens[1], 1)
    lie_basis = construct_lie_basis_general(gens)
    
    #@assert check_group_implementability(target, lie_basis) "Target is not implementable with given generators"
    params = Params(
        -im*drift,
        -im*control,
        -im*Vector(diag(control)),
        tmin,
        tmax,
        time_coeff,
        coset_tol,
        abs_tol,
        rel_tol,
        beta,
        dim,
        Matrix{T}(undef, dim, dim),
        Matrix{T}(undef, dim, dim),        
        Matrix{T}(undef, dim, dim)       
    )

    p_basis = lie_basis[2:end] # exclude control subalgebra from orthogonal complement
    M0 = zeros(ComplexF64, size(p_basis[1])...)

    dself = distance_to_target_coset(Matrix(target), target, params)
    U = Matrix(target)
    @assert norm(U' * U - I(dim), Inf) ≤ 1e-10 "Target not unitary"
    @assert dself < params.coset_tol "Target not in target coset within tolerance"

    function single_shoot(m)
        X0 = zeros(T, 2*dim^2)
        @views P0 = reshape(view(X0, 1:dim^2), dim, dim)
        copyto!(P0, I)
        build_M!(M0, m, p_basis)
        @views reshape(X0[dim^2 + 1: 2*dim^2], dim, dim) .= M0

        prob = ODEProblem(f!, X0, (0.0, params.tmax), params)
        # TODO: Choose propagator where we can set the time-step
        sol = solve(prob, Tsit5(); abstol=params.abs_tol, reltol=params.rel_tol,
        saveat=range(0.0, params.tmax, length=n_saves))
        return sol
    end

    function distance_at_time(t, sol)
        Xt = sol(t)
        @views P_rot = reshape(view(Xt, 1:dim^2), dim, dim)
        return distance_to_target_coset(P_rot, target, params)
    end

    function evaluate_single_shoot(sol, params)
        
        # func(t) = distance_at_time(t, sol)
        # res = optimize(func, params.tmin, params.tmax)
        # dmin, tstar = Optim.minimum(res), Optim.minimizer(res)

        ts = range(params.tmin, params.tmax, n_saves)
        vals = [distance_at_time(t, sol) for t in ts]
        dmin = minimum(vals)
        tstar = ts[argmin(vals)]

        return dmin, tstar
    end

    function objective(m)
        
        sol = single_shoot(m)
        dmin, tstar = evaluate_single_shoot(sol, params)
    # function objective(m)
    #     sol = single_shoot(m)
    #     ts = range(params.tmin, params.tmax, n_saves)
    #     vals = [distance_at_time(t, sol) for t in ts]
    #     J = sum(exp.(-params.beta * vals)) / length(vals)
    #     return -log(J)/params.beta
    # end
        return dmin + params.time_coeff * tstar
    end

    m0 = randn(length(p_basis))
    result =minimize(objective, m0, 0.5; maxiter=50, verbosity=1)
    m_best = xbest(result)
    sol_best = single_shoot(m_best)
    dmin, tstar = evaluate_single_shoot(sol_best, params)

    ustar = sol_best(tstar)
    P_rot_best = reshape(view(ustar, 1:dim^2), dim, dim)

    println("Best m0: $m_best, min distance: $dmin, time (in π): $(tstar/pi)")
    # println("Propagator in rot frame at t=0:")
    # display(reshape(view(sol_best(0), 1:dim^2), dim, dim))
    println("Propagator in rot frame at tstar:")
    display(P_rot_best)
    ts = range(params.tmin, params.tmax; length=n_saves)
    ds_rot = [distance_at_time(t, sol_best) for t in ts]
    p = plot(ts/pi, ds_rot, label="Rot")
    xlabel!(p, "Time in π")
    ylabel!(p, "Distance to target coset")
    display(p)
    savefig(p, "Two qutrit best.png")
end

main_function()




