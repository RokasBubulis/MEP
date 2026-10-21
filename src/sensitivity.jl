# include("structs.jl")

function costate_sensitivity_drift!(res, algebra::Algebra, stor::Storage, M_lie::Vector{Float64})
    # TODO verify that res is cleanly overwritten
    d = size(algebra.lie_basis, 1)    
    f_contract_H = reshape(transpose(stor.H_opt_prime_lie) * reshape(algebra.structure_tensor, d, :), d,d)
    f_contract_HM = 1/stor.overlap_Hdprime_M * reshape(transpose(M_lie) * f_contract_H, d, 1)
    res .= f_contract_HM * transpose(stor.H_opt_prime_lie)
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

function infidelity(U::AbstractMatrix, tar::TargetContainer, algebra::Algebra, stor::Storage)
    n = size(tar.adjoint_target, 1)
    mul!(stor.tmp_3levels, U, tar.adjoint_target)

    for i in eachindex(stor.tmp_diag_3levels)
        stor.tmp_diag_3levels[i] = stor.tmp_3levels[i, i]
    end
    tmp_diag       = stor.tmp_diag_3levels
    im_control_vec = algebra.im_control_vec_3levels
    exp_buffer     = stor.exp_buffer_dist_3level
    # d(β) = 1 - |⟨exp(iβΣZ), diag(U U_tar')⟩| / n, minimised over β
    dist = function (β)
        @. exp_buffer = exp(β * im_control_vec)
        return 1 - (1/n) * abs(dot(exp_buffer, tmp_diag))
    end

    res = optimize(dist, -π, π, Brent())
    stor.W = conjugate(exp(res.minimizer * algebra.lie_basis[1]) * tar.target)
    
    return res.minimum
end 

function infidelity_gradient!(grad::Vector{Float64}, algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    n = size(algebra.im_drift, 1)
    for μ in 1:d
        U_deriv_m_mu = sum(stor.B[μ, ν] * algebra.lie_basis[ν] for ν in 1:d) * stor.U
        grad[μ] .= real(dot(-1/n*transpose(stor.W), U_deriv_m_mu))
    end 
end 