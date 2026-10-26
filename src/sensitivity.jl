# include("structs.jl")

function costate_sensitivity_drift!(res, algebra::Algebra, stor::Storage, M_lie::Vector{Float64})
    # TODO verify that res is cleanly overwritten
    d = size(algebra.lie_basis, 1)    
    f_contract_Hprime = reshape(transpose(stor.H_opt_prime_lie) * reshape(algebra.structure_tensor, d, :), d,d)
    f_contract_Hprime_M = reshape(transpose(M_lie) * f_contract_Hprime, d, 1)
    res .= 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie * transpose(f_contract_Hprime_M)
    res .+= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d)
    return nothing
end

function state_sensitivity_source!(res, algebra::Algebra, stor::Storage)
    res .= 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie * transpose(stor.H_opt_prime_lie)
    return nothing
end 

function state_sensitivity_drift!(res, algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    res .= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d)
    return nothing
end 

function infidelity(U::AbstractMatrix, algebra::Algebra, stor::Storage, tar::TargetContainer,)
    n = size(tar.adjoint_target, 1)
    mul!(stor.tmp_3levels, U, tar.adjoint_target)

    for i in eachindex(stor.tmp_diag_3levels)
        stor.tmp_diag_3levels[i] = stor.tmp_3levels[i, i]
    end
    tmp_diag       = stor.tmp_diag_3levels
    im_control_vec = algebra.im_control_vec_3levels
    exp_buffer     = stor.exp_buffer_dist_3levels
    # d(β) = 1 - |⟨exp(iβΣZ), diag(U U_tar')⟩| / n, minimised over β
    dist = function (β)
        @. exp_buffer = exp(β * im_control_vec)
        return 1 - (1/n) * abs(dot(exp_buffer, tmp_diag))
    end

    res = optimize(dist, -π, π, Brent())
    # instead of full W, store only optimal β and construct W as needed
    #stor.W = conj(exp(res.minimizer * Matrix(algebra.lie_basis[1])) * tar.target)
    stor.β_opt = res.minimizer
    return res.minimum
end 

function dU_dm_mu(μ::Int, algebra::Algebra, tracker::CostOptimumTracker)
    d = length(algebra.lie_basis)
    grad = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν] for ν in 1:d) * tracker.U_t_star
    return grad 
end 

function grad_infidelity!(grad::Vector{Float64}, algebra::Algebra, tracker::CostOptimumTracker)
    d = size(algebra.lie_basis, 1)
    n = size(algebra.im_drift, 1)
    for μ in 1:d
        U_deriv_m_mu = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν] for ν in 1:d) * tracker.U_t_star
        grad[μ] = real(dot(-1/n*transpose(tracker.W_t_star), U_deriv_m_mu))
    end 
end 

function fd_grad(phi, m; dm = 1e-6)
    g = similar(m, Float64)
    for i in eachindex(m)
        mp = copy(m); mp[i] += dm
        mm = copy(m); mm[i] -= dm
        g[i] = (phi(mp) - phi(mm)) / (2dm)
    end
    return g
end