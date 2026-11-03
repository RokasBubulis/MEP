# include("structs.jl")

function structure_contract!(res, f::Array{Float64, 3}, x::Vector{Float64})
    # res[j,k] = Σ_i x_i f[i,j,k], written into res without creating temporary arrays
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
    # D = 1/h H' [H', M]ᵀ + F,  h = stor.overlap_Hdprime_M
    # F (state_sensitivity_drift!) must already be computed in stor.F
    lie_bracket_coeffs!(stor.bracket_lie, algebra.structure_tensor, stor.H_opt_prime_lie, M_lie)   # [H', M]
    s = 1/stor.overlap_Hdprime_M
    Hp = stor.H_opt_prime_lie
    @inbounds for k in eachindex(Hp), a in eachindex(Hp)
        res[a, k] = s * Hp[a] * stor.bracket_lie[k] + stor.F[a, k]
    end
    return nothing
end

function state_sensitivity_source!(res, stor::Storage)
    # E = 1/h H' H'ᵀ
    s = 1/stor.overlap_Hdprime_M
    Hp = stor.H_opt_prime_lie
    @inbounds for k in eachindex(Hp), a in eachindex(Hp)
        res[a, k] = s * Hp[a] * Hp[k]
    end
    return nothing
end

function state_sensitivity_drift!(res, algebra::Algebra, stor::Storage)
    # F[j,k] = Σ_i H_i f[i,j,k]
    structure_contract!(res, algebra.structure_tensor, stor.H_opt_lie)
    return nothing
end

function infidelity(U::AbstractMatrix, algebra::Algebra, stor::Storage, tar::TargetContainer,)
    # a logical (2^N x 2^N) target selects the cost on the logical subspace: U is projected first
    n = size(tar.adjoint_target, 1)

    if size(U, 1) != n
        project_from_3_to_2_levels_general!(stor.U_2levels, U, algebra, stor)
        mul!(stor.tmp_2levels, stor.U_2levels, tar.adjoint_target)
        for i in eachindex(stor.tmp_diag_2levels)
            stor.tmp_diag_2levels[i] = stor.tmp_2levels[i, i]
        end
        tmp_diag       = stor.tmp_diag_2levels
        im_control_vec = algebra.im_control_vec_2levels
        exp_buffer     = stor.exp_buffer_dist_2levels
    else
        mul!(stor.tmp_3levels, U, tar.adjoint_target)
        for i in eachindex(stor.tmp_diag_3levels)
            stor.tmp_diag_3levels[i] = stor.tmp_3levels[i, i]
        end
        tmp_diag       = stor.tmp_diag_3levels
        im_control_vec = algebra.im_control_vec_3levels
        exp_buffer     = stor.exp_buffer_dist_3levels
    end
    # d(β) = 1 - |⟨exp(iβΣZ), diag(U U_tar')⟩| / n, minimised over β
    dist = function (β)
        @. exp_buffer = exp(β * im_control_vec)
        return 1 - (1/n) * abs(dot(exp_buffer, tmp_diag))
    end

    res = optimize(dist, -π, π, Brent())
    # instead of full W, store only optimal β and construct W as needed
    stor.β_opt = res.minimizer
    return res.minimum
end 

function conj_rotated_target!(W, β, algebra::Algebra, tar::TargetContainer)
    # W = conj(exp(iβC) S) in the space of the target (C is diagonal; logical C if the target is logical)
    im_control_vec = size(tar.target, 1) == length(algebra.im_control_vec_3levels) ? algebra.im_control_vec_3levels : algebra.im_control_vec_2levels
    W .= conj.(exp.(β .* im_control_vec) .* tar.target)
    return nothing
end

# projection onto the logical subspace when the target (W_t_star) is logical, identity otherwise
project_to_target_space(X, W, algebra::Algebra) = size(W, 1) == size(X, 1) ? X : project_from_3_to_2_levels(X, algebra.ryd_to_logic_conv_mat)

function dU_dm_mu(μ::Int, algebra::Algebra, tracker::CostOptimumTracker)
    d = length(algebra.lie_basis)
    grad = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν] for ν in 1:d) * tracker.U_t_star
    return grad 
end 

function pad_to_full_space!(W_pad, W, algebra::Algebra)
    # W_pad = Π W Π†: W on the logical rows/columns and zero elsewhere (W_pad = W if W is already full size)
    # then  sum(W .* P(X)) = sum(W_pad .* X)  for any full-space X, with P(X) = Π† X Π
    if size(W) == size(W_pad)
        W_pad .= W
    else
        Π = algebra.ryd_to_logic_conv_mat
        mul!(W_pad, Π * W, adjoint(Π))
    end
    return nothing
end

function trace_of_product(L::SparseMatrixCSC, P::Matrix)
    # tr(L P) = Σ_ab L_ab P_ba, summed over the nonzero entries of the sparse L only
    rows = rowvals(L)
    vals = nonzeros(L)
    s = zero(ComplexF64)
    @inbounds for b in 1:size(L, 2), idx in nzrange(L, b)
        s += vals[idx] * P[b, rows[idx]]
    end
    return s
end

function grad_infidelity!(grad::Vector{Float64}, algebra::Algebra, tracker::CostOptimumTracker)
    # φ = 1 - |z|/n,  z = <S', P(U)> = tr(S'† P(U)),  S' = exp(iβ_opt C) S  (β_opt fixed by envelope theorem)
    # ∂φ/∂m_μ = -(1/n) Re( conj(z)/|z| · <S', P(∂U/∂m_μ)> )
    # P = projection onto the logical subspace if the target is logical (n = 2^N), identity otherwise (n = 3^N)
    #
    # Same formula, evaluated without building P(∂U/∂m_μ) for every μ:
    #   <S', P(X)> = sum(W_pad .* X) = tr(X W_padᵀ),   ∂U/∂m_μ = Σ_ν B[μ,ν] L_ν U  (see dU_dm_mu)
    #   ⇒ <S', P(∂U/∂m_μ)> = Σ_ν B[μ,ν] g_ν,   g_ν = tr(L_ν U W_padᵀ),   z = tr(U W_padᵀ)
    n = size(tracker.W_t_star, 1)
    P = tracker.P
    mul!(P, tracker.U_t_star, transpose(tracker.W_pad))      # P = U W_padᵀ
    z = tr(P)
    c = conj(z)/abs(z)
    for (ν, L) in enumerate(algebra.lie_basis)
        tracker.g_lie[ν] = real(c * trace_of_product(L, P))
    end
    mul!(grad, tracker.B_t_star, tracker.g_lie, -1/n, 0.0)   # grad = -(1/n) B g   (B is real)
    return nothing
end

function dphi_dT(algebra::Algebra, tracker::CostOptimumTracker)
    # dU/dT = H_opt(T) U  ⇒  ∂φ/∂T = -(1/n) Re( conj(z)/|z| · <S', P(H_opt(T) U)> )
    # with P = U W_padᵀ:  z = tr(P),  <S', P(H U)> = tr(H P)
    n = size(tracker.W_t_star, 1)
    P = tracker.P
    mul!(P, tracker.U_t_star, transpose(tracker.W_pad))
    z = tr(P)
    c = conj(z)/abs(z)
    H = tracker.H_t_star
    s = zero(ComplexF64)
    @inbounds for b in axes(P, 2), a in axes(P, 1)
        s += H[a, b] * P[b, a]
    end
    return real(-c/n * s)
end

function project_from_3_to_2_levels_general!(res, mat, algebra, stor)
    # U^(2) = Φ * Ψ' * U^(3) * Ψ * Φ' 
    # where Ψ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0,0), (0,1,0)}
    # Φ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0), (0,1)}
    # where n is the number of particles 
    # this allows projection from 3 level operator to 2 where |r> states are excluded

    mul!(stor.tmp_logical1, mat, algebra.ryd_to_logic_conv_mat)
    mul!(res, algebra.adj_ryd_to_logic_conv_mat, stor.tmp_logical1)
    return nothing
end