function structure_contract!(res, f::Array{Float64, 3}, x::Vector{Float64})
    d = length(x)
    @inbounds for k in 1:d, j in 1:d
        s = 0.0
        for i in 1:d
            s += x[i] * f[i, j, k]
        end
        res[j, k] = s
    end
    return nothing
end

function costate_sensitivity_drift!(res, algebra::Algebra, stor::Storage, M_lie::Vector{Float64})
    lie_bracket_coeffs!(stor.bracket_lie, algebra.structure_tensor, stor.H_opt_prime_lie, M_lie)
    s = 1/stor.overlap_Hdprime_M
    Hp = stor.H_opt_prime_lie
    @inbounds for k in eachindex(Hp), a in eachindex(Hp)
        res[a, k] = s * Hp[a] * stor.bracket_lie[k] + stor.F[a, k]
    end
    return nothing
end

function state_sensitivity_source!(res, stor::Storage)
    s = 1/stor.overlap_Hdprime_M
    Hp = stor.H_opt_prime_lie
    @inbounds for k in eachindex(Hp), a in eachindex(Hp)
        res[a, k] = s * Hp[a] * Hp[k]
    end
    return nothing
end

function state_sensitivity_drift!(res, algebra::Algebra, stor::Storage)
    structure_contract!(res, algebra.structure_tensor, stor.H_opt_lie)
    return nothing
end

# The target amplitudes on the invariant subspaces, indexed by r = 0 … N:
#   r = 0 is the dark |0…0⟩ state (a_0 = 1 identically: every Lie element annihilates it)
#   r ≥ 1 is the weight-r subspace |1…1⟩|0…0⟩, whose propagator acts as e^{-irθ} under
#          Z(θ)⊗…⊗Z(θ) = exp(iθ Σ_j |1⟩⟨1|_j), and which carries C(N, r) logical states
# The gate is 1 on the all-ones class (k = 1) or the all-zeros class (k = 0) and e^{-iϕ} elsewhere.
#
# The dark amplitude is kept even though it is constant: it is what makes the error
# non-degenerate. Without it the free θ can rephase every term of z, |z| saturates the
# normaliser at U = I, and the identity scores a perfect gate.
function target_diag(θ::Real, k::Int, ϕ::Float64, N::Int)
    return [binomial(N, r) * exp(-im * r * θ) * ((k == 1 ? r == N : r == 0) ? 1.0 : exp(-im * ϕ))
            for r ∈ 0:N]
end

# # τ_r of the target, from N, k and ϕ alone (index 1 is the dark |0…0⟩ state)
# function target_amplitudes!(stor::Storage, algebra::Algebra, tar::TargetContainer)
#     N = algebra.n_particles
#     @inbounds for r in 0:N
#         stor.target_amplitude[r + 1] = ((tar.k == 1 ? r == N : r == 0) ? 1.0 + 0im : cis(-tar.ϕ))
#     end
#     return nothing
# end

# a_r = (U^(r))[1,1] on the N invariant subspaces, preceded by a_0 = 1 for the dark |0…0⟩ state
state_diag(U::Vector{Matrix{ComplexF64}}) = [1.0 + 0im; [U[λ][1, 1] for λ ∈ eachindex(U)]]

function cost(U::Vector{Matrix{ComplexF64}}, T::Real, algebra::Algebra, stor::Storage, k::Int, ϕ::Float64, time_weight::Real)
    N = algebra.n_particles
    ket = state_diag(U)
    cost_f = function (θ)
        return 1 - abs(dot(target_diag(θ, k, ϕ, N), ket)) / 2^N + time_weight * T
    end
    # Φ(θ) is 2π-periodic and can have several local minima: coarse grid first, then Brent around the best point
    Φs = range(-π, π, length=65)
    Φ0 = Φs[argmin(cost_f.(Φs))]
    res = optimize(cost_f, Φ0 - 2π/64, Φ0 + 2π/64, Brent())
    stor.phase_opt = res.minimizer
    return res.minimum
end

function cost_grad!(grad::Vector{Float64}, algebra::Algebra, phase_opt::Float64, tracker::PropagationOutput, k::Int, ϕ::Float64, time_weight::Real)
    N = algebra.n_particles
    d = length(algebra.lie_basis)
    bra = target_diag(phase_opt, k, ϕ, N)          # length N + 1, index 1 is the dark state
    ket = tracker.a_t                              # length N + 1, a_r = (U^(r))[1,1], a_0 = 1

    # ∂φ/∂a_r  (θ_opt fixed by the envelope theorem)
    z = dot(bra, ket)
    da = -conj(z) / abs(z) / 2^N .* conj.(bra)

    op_full, op_λ, tmp = tracker.op_full, tracker.op_λ, tracker.iso_tmp

    # dΦ/dm^μ
    for μ ∈ 1:d
        # the dark |0…0⟩ row stays 0: every Lie element annihilates it, so da[1] has nothing to act on
        ket_dm = zeros(ComplexF64, N + 1)
        # B_μ = Σ_ν B_t[μ,ν] L_ν ∈ g, assembled in the full 3^N representation
        fill!(op_full, zero(ComplexF64))
        @inbounds for ν ∈ 1:d
            c = tracker.B_t_star[μ, ν]
            c == 0 && continue
            axpy!(c, algebra.lie_basis[ν], op_full)
        end
        for λ ∈ 1:N
            # B_μ^(λ) = V_λ' B_μ V_λ ∈ gl(C^2), then dU^(λ)/dm^μ = B_μ^(λ) U^(λ)
            apply_λth_isometry_transformation!(op_λ, op_full, algebra.isometries[λ], tmp)
            ket_dm[λ + 1] = (op_λ * tracker.U_t_star[λ])[1, 1]
        end
        grad[μ] = real(sum(da .* ket_dm))   # plain product: dot would conjugate da
    end

    # dΦ/dt
    ket_dt = zeros(ComplexF64, N + 1)
    fill!(op_full, zero(ComplexF64))
    @inbounds for ν ∈ 1:d
        c = tracker.H_t_star[ν]
        c == 0 && continue
        axpy!(c, algebra.lie_basis[ν], op_full)
    end
    for λ ∈ 1:N
        apply_λth_isometry_transformation!(op_λ, op_full, algebra.isometries[λ], tmp)
        ket_dt[λ + 1] = (op_λ * tracker.U_t_star[λ])[1, 1]
    end
    grad[d + 1] = real(sum(da .* ket_dt)) + time_weight
    return nothing
end

# ── the names src/optimise.jl and the dev tests already call ─────────────────────────────────
# your cost_grad! writes dΦ/dm^μ into grad[1:d] and dΦ/dT into grad[d+1]; these two adapters
# expose it under the names optimise.jl uses, without a second implementation.
function grad_infidelity!(grad::Vector{Float64}, algebra::Algebra, tracker::PropagationOutput)
    g = zeros(Float64, length(algebra.lie_basis) + 1)
    cost_grad!(g, algebra, tracker.phase_opt, tracker, tracker.target_k, tracker.target_ϕ, 0.0)
    copyto!(grad, view(g, 1:length(grad)))
    return nothing
end

function dphi_dT(algebra::Algebra, tracker::PropagationOutput)
    g = zeros(Float64, length(algebra.lie_basis) + 1)
    cost_grad!(g, algebra, tracker.phase_opt, tracker, tracker.target_k, tracker.target_ϕ, 0.0)
    return g[end]
end
