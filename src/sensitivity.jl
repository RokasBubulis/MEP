# include("structs.jl")

# using LinearAlgebra

# function χ_dot!(algebra::Algebra, stor::Storage, u)
#     n = size(algebra.lie_basis, 1)
#     stor.dχ .= 0
#     for μ, ν in 1:n 
#         for α, β, γ in 1:n
#             stor.dχ[μ, ν] += -1/stor.a * stor.H_opt_prime_lie[α] * u.x[β] * algebra.structure_tensor[α, β, ν] * stor.H_opt_prime_lie[γ] * stor.χ[μ, γ] + stor.H_opt_lie[α] * stor.χ[μ, β] * algebra.structure_tensor[α, β, ν]
#         end 
#     end 
#     return nothing
# end 

# function ω_dot!(algebra::Algebra, stor::Storage)
#     d = size(algebra.lie_basis, 1)

#     # Hchi[μ] = Σ_ν H_opt_prime_lie[ν] * χ[μ, ν]  ==  χ * H_opt_prime_lie
#     mul!(stor.Hchi, stor.χ, stor.H_opt_prime_lie)

#     # H_opt_prime * U is independent of μ
#     mul!(stor.Hprime_U, stor.H_opt_prime, stor.U)

#     inv_a = -1 / stor.a
#     @inbounds for μ in 1:d
#         dωμ = view(stor.dω, :, :, μ)
#         ωμ  = view(stor.ω,  :, :, μ)
#         mul!(dωμ, stor.H_opt, ωμ)  # Σ_k H_opt[i,k] ω[k,j,μ]
#         axpy!(inv_a * stor.Hchi[μ], stor.Hprime_U, dωμ)  # source term
#     end
#     return nothing
# end 


function chi_dot!(algebra::Algebra, stor::Storage, M_lie::Vector{Float64})
    d = size(algebra.lie_basis, 1)    
    f_contract_H = reshape(transpose(stor.H_opt_prime_lie) * reshape(algebra.structure_tensor, d, :), d,d)
    f_contract_HM = 1/stor.overlap_Hdprime_M * reshape(transpose(M_lie) * f_contract_H, d, 1)
    stor.dχ .= f_contract_HM * transpose(stor.H_opt_prime_lie)
    stor.dχ .+= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d)
end

function b_dot!(algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    stor.db .= 1/stor.overlap_Hdprime_M * stor.χ * stor.H_opt_prime_lie * transpose(stor.H_opt_prime_lie)
    stor.db .+= reshape(transpose(stor.H_opt_lie) * reshape(algebra.structure_tensor, d, :), d, d) * stor.b
end 

