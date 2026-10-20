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


function b_dot_elementwise!(algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    for μ in 1:d, ν in 1:d
        for α in 1:d, β in 1:d, γ in 1:d
            stor.db[μ, ν] = 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie[α] * stor.χ[μ, α] * stor.H_opt_prime_lie[ν] + stor.H_opt_lie[β] * stor.b[μ, γ] * algebra.structure_tensor[β, γ, ν]
        end 
    end
end 

function b_dot_elementwise_inverse_order!(algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    for μ in 1:d, ν in 1:d
        for α in 1:d, β in 1:d, γ in 1:d
            stor.db[μ, ν] = 1/stor.overlap_Hdprime_M * stor.H_opt_prime_lie[α] * stor.χ[μ, α] * stor.H_opt_prime_lie[ν] + stor.H_opt_lie[β] * stor.b[μ, γ] * algebra.structure_tensor[β, γ, ν]
        end 
    end
end 
hfdkjfhdfdskjfsiwebisik dgffuweuffffdfdsf
function b_dot_matrix!(algebra::Algebra, stor::Storage)
    d = size(algebra.lie_basis, 1)
    mat = reshape(reshape(algebra.structure_tensor, d*d, d) * stor.H_opt_lie, d, d)
    stor.db .= 1/stor.overlap_Hdprime_M * stor.χ * stor.H_opt_prime_lie * transpose(stor.H_opt_prime_lie) + stor.b * mat
end
