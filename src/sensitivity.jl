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

function target_diag(θ::Real, k::Int, ϕ::Float64, N::Int)
    return [binomial(N, r) * exp(-im * r * θ) * (r == 0 ? 1.0 : exp(-im * ϕ))
            for r ∈ 0:N]
end

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

    op_λ = tracker.op_λ
    B_row = tracker.b_row

    # dΦ/dm^μ
    for μ ∈ 1:d
        # the dark |0…0⟩ row stays 0: every Lie element annihilates it, so da[1] has nothing to act on
        ket_dm = zeros(ComplexF64, N + 1)

        # B_μ^(λ) = Σ_ν (A^(λ) b_μ)_ν L̃^(λ)_ν ∈ gl(C^2), then dU^(λ)/dm^μ = B_μ^(λ) U^(λ)
        for λ ∈ 1:N
            copyto!(B_row, view(tracker.B_t_star, μ, :))
            assemble_invariant!(op_λ, tracker.tmp_coeff,
                                algebra.maps_lie_to_compr_eff_subspace[λ],
                                algebra.compr_eff_subspaces_bases[λ], B_row)
            ket_dm[λ + 1] = (op_λ * tracker.U_t_star[λ])[1, 1]
        end
        grad[μ] = real(sum(da .* ket_dm))   # plain product: dot would conjugate da
    end

    # dΦ/dt
    ket_dt = zeros(ComplexF64, N + 1)
    for λ ∈ 1:N
        assemble_invariant!(op_λ, tracker.tmp_coeff,
                            algebra.maps_lie_to_compr_eff_subspace[λ],
                            algebra.compr_eff_subspaces_bases[λ], tracker.H_t_star)
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
