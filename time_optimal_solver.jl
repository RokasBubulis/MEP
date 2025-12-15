using DifferentialEquations
using Optim
using FastExpm
using ComponentArrays
using LinearAlgebra
using SparseArrays

include("generators.jl")
include("lie_algebra.jl")
include("implementability.jl")

# Commutator
br(A, B) = A*B - B*A

# Drift + detuning control
H_delta(delta, p) = p.H0 + delta * p.L

# Parameter struct
struct Params
    H0::Matrix{ComplexF64}
    L::Matrix{ComplexF64}
    tol::Float64
    u_min::Float64
    u_max::Float64
    ε::Float64
end
# ─────────────────────────────────────────────────────────────
# PMP-inspired continuous control law: δ(t) from M(t)
# δ(t) = clamp((1/ε) * Re(tr(M L)), u_min, u_max)
# ─────────────────────────────────────────────────────────────

@inline function control_delta(M, p)
    σ = real(tr(M * p.L))  
    # δ_unclipped = (1 / p.ε) * σ        # smooth approx to bang–bang
    return clamp(σ, p.u_min, p.u_max)
end

# ─────────────────────────────────────────────────────────────
# Distance to target coset:  min_δ || target† P - exp(δ L_diag) ||
# (still uses a 1D optimization, but called only in the callback)
# ─────────────────────────────────────────────────────────────

function get_distance_to_target_coset(P_opt, target, p)
    λ = diag(p.L)                 # eigenvalues of L on diagonal
    # A = adjoint(target) * P_opt   # relative unitary
    A = P_opt * adjoint(target)

    function cost(delta_vec)
        δ = delta_vec[1]
        D = spdiagm(0 => exp.(δ .* λ))
        return norm(A - D)        # 2-norm (spectral) by default
    end

    δ0 = [0.0]
    result = optimize(cost, δ0, NelderMead())
    δ_opt = Optim.minimizer(result)[1]

    return cost([δ_opt])          # minimum distance
end

# ─────────────────────────────────────────────────────────────
# ODE RHS: X = (P, M),   Ẋ = (H P, [H,M]),  H = H0 + δ(t)L
# δ(t) is chosen from M(t) via control_delta (no inner optimize!)
# ─────────────────────────────────────────────────────────────

function f!(dX, X, p, t)
    P, M = X.P, X.M

    δ = control_delta(M, p)
    H = p.H0 .+ δ .* p.L

    dX.P .= H * P
    dX.M .= br(H, M)
end

# ─────────────────────────────────────────────────────────────
# Build M from coefficients m in basis p_basis
# ─────────────────────────────────────────────────────────────

function build_M(m, p_basis)
    M = zeros(eltype(p_basis[1]), size(p_basis[1]))
    for (i, m_coeff) in enumerate(m)
        @assert size(p_basis[i]) == size(M)
        M .+= m_coeff * p_basis[i]
    end
    return M
end

# ─────────────────────────────────────────────────────────────
# Main: compute_optimal_time
#   gens   : [H_drift_basis_element, H_control_basis_element, ...]
#   target : target unitary
#
# Outer optimization over initial M (parameters m).
# ─────────────────────────────────────────────────────────────
function compute_optimal_time(gens, target; tmax = 20.0,
                              coset_tolerance = 1e-3,
                              penalty = 10.0,
                              u_min = -1.0,
                              u_max = 1.0,
                              ε = 0.05)

    # Construct Lie basis and check implementability
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"

    # Choose a basis for M (you can adapt this if you like)
    # Here: assume controls correspond to basis element #2; use others for M
    # p_basis = [[lie_basis[1]]; lie_basis[3:end]]
    p_basis = lie_basis

    # Parameters for dynamics
    params = Params(
        -im * gens[1],   # drift
        -im * gens[2],   # control generator
        coset_tolerance,
        u_min, u_max, ε,
    )

    # Sanity check: different M => different H (should not collapse)
    M1 = build_M(rand(length(p_basis)) .* 2 .- 1, p_basis)
    M2 = build_M(rand(length(p_basis)) .* 2 .- 1, p_basis)
    H1 = H_delta(control_delta(M1, params), params)
    H2 = H_delta(control_delta(M2, params), params)
    @show norm(H1 - H2)

    # Single trajectory given initial m
    function single_sol_run(m; tmax=tmax)
        M0 = build_M(m, p_basis)
        # Identity of appropriate size (adapt if you need kron(id,id))
        d = size(gens[1], 1)
        P0 = Matrix(I, d, d)
        X0 = ComponentArray(P = P0, M = M0)

        tspan = (0.0, tmax)
        prob = ODEProblem(f!, X0, tspan, params)

        min_dist = Ref(Inf)
        hit_time = Ref(nothing)

        # Check distance every N iterations (reduce for speed)
        condition(u, t, integrator) = (integrator.iter % 20 == 0)

        function affect!(integrator)
            X = integrator.u
            P, M = X.P, X.M
            p_loc = integrator.p

            dist = get_distance_to_target_coset(P, target, p_loc)
            if dist < min_dist[]
                min_dist[] = dist
            end

            if dist < real(p_loc.tol) && hit_time[] === nothing
                hit_time[] = integrator.t
                terminate!(integrator)
            end
        end

        cb = DiscreteCallback(condition, affect!)
        sol = solve(prob, callback = cb)

        return hit_time[], min_dist[]
    end

    # Outer objective over initial m (coefficients of M0 in p_basis)
    function objective(m)
        hit_time, min_dist = single_sol_run(m)
        println("Evaluating at m = $m : hit_T = $hit_time, min_dist = $min_dist")
        if hit_time !== nothing
            return hit_time
        else
            return tmax + penalty * min_dist
        end
    end

    # Initial guess for m
    m0 = rand(length(p_basis)) .* 2 .- 1

    # Outer optimization (no nested LBFGS inside ODE anymore!)
    result = optimize(objective, m0, NelderMead())
    m_opt = Optim.minimizer(result)
    T_opt = Optim.minimum(result)

    println("Optimal m: ", m_opt)
    println("Optimal time T_opt: ", T_opt)
    println(result)

    return m_opt, T_opt, result
end