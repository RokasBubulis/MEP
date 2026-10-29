# include("structs.jl")

function costate_sensitivity_drift!(res, algebra::Algebra, stor::Storage, M_lie::Vector{Float64})
    d = size(algebra.lie_basis, 1)    
    f_contract_Hprime = reshape(transpose(stor.H_opt_prime_lie) * reshape(algebra.structure_tensor, d, :), d,d)
    f_contract_Hprime_M = reshape(transpose(M_lie) * f_contract_Hprime, d, 1)
    res .= 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie * transpose(f_contract_Hprime_M)
    res .+= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d)
    return nothing
end

function state_sensitivity_source!(res, stor::Storage)
    res .= 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie * transpose(stor.H_opt_prime_lie)
    return nothing
end 

function state_sensitivity_drift!(res, algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    res .= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d)
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

function grad_infidelity!(grad::Vector{Float64}, algebra::Algebra, tracker::CostOptimumTracker)
    # φ = 1 - |z|/n,  z = <S', P(U)> = tr(S'† P(U)),  S' = exp(iβ_opt C) S  (β_opt fixed by envelope theorem)
    # ∂φ/∂m_μ = -(1/n) Re( conj(z)/|z| · <S', P(∂U/∂m_μ)> )
    # P = projection onto the logical subspace if the target is logical (n = 2^N), identity otherwise (n = 3^N)
    d = length(algebra.lie_basis)
    W = tracker.W_t_star                      # W_t_star = conj(S')  ⇒  z = <S', P(U)>
    n = size(W, 1)
    z = sum(W .* project_to_target_space(tracker.U_t_star, W, algebra))
    c = conj(z)/abs(z)
    for μ in 1:d
        grad[μ] = real(-c/n * sum(W .* project_to_target_space(dU_dm_mu(μ, algebra, tracker), W, algebra)))
    end
    return nothing
end

function dphi_dT(algebra::Algebra, tracker::CostOptimumTracker)
    # dU/dT = H_opt(T) U  ⇒  ∂φ/∂T = -(1/n) Re( conj(z)/|z| · <S', P(H_opt(T) U)> )
    W = tracker.W_t_star                      # W_t_star = conj(S')  ⇒  z = <S', P(U)>
    n = size(W, 1)
    U = tracker.U_t_star
    z = sum(W .* project_to_target_space(U, W, algebra))
    c = conj(z)/abs(z)
    return real(-c/n * sum(W .* project_to_target_space(tracker.H_t_star * U, W, algebra)))
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