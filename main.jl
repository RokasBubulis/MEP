include("generators.jl")
include("lie_algebra.jl")

using Plots, BenchmarkTools

function check_target(target::SparseMatrixCSC{ComplexF64, Int}, system_basis::Vector{SparseMatrixCSC{ComplexF64, Int}})

    reconstructed_target = spzeros(ComplexF64, size(target))
    for i in eachindex(system_basis)
        coeff = tr(adjoint(system_basis[i]) * target)
        reconstructed_target += coeff * system_basis[i]
    end

    residual = reconstructed_target - target
    return norm(residual), residual
end

function construct_target(n_qubits::Int)
    I3 = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    target = copy(I3)
    for n in 2:n_qubits
        target = kron(target, I3)
    end
    target[3^n_qubits, 3^n_qubits] = -1.0
    return target
end

function construct_subgroup_basis(lie_basis::Vector{SparseMatrixCSC{ComplexF64, Int}}, product_depth::Int)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[]
    try_add_orthonormal!(basis_elements, lie_basis[1]^0) # add identity

    for b in lie_basis
        try_add_orthonormal!(basis_elements, b)
    end
    
    if product_depth > 1
        for d in 2:product_depth
            for tuple in Iterators.product(ntuple(_ -> lie_basis, d)...)
                try_add_orthonormal!(basis_elements, foldl(*, tuple))
            end
        end
    end
    return basis_elements
end

# a = [1, 2, 3]
# b = 3
# for tup in Iterators.product(ntuple(_ -> a, b)...)
#     println(tup)
# end

n_qubits = 2
commutation_depth = 10
product_depth = 4
target = construct_target(n_qubits)
generators = construct_Ryd_generators(n_qubits)
lie_basis = @btime construct_lie_basis_fast(generators, commutation_depth)
subgroup_basis = @btime construct_subgroup_basis(lie_basis, product_depth)
residual_norm, residual = check_target(target, subgroup_basis)
println("Lie algebra dim: $(length(lie_basis)), Lie subgroup dim: $(length(subgroup_basis)), Residual norm: $residual_norm")


# qubit_lst = 2:3
# product_depth = 1:4
# residuals = zeros(length(qubit_lst), length(product_depth))

# for (i, n_qubits) in enumerate(qubit_lst)

#     println("n qubits: $n_qubits")
#     local generators, target, lie_basis
#     generators = construct_Ryd_generators(n_qubits)
#     target = construct_target(n_qubits)
#     lie_basis = construct_lie_basis_fast(generators, commutation_depth)

#     for (j, product_depth) in enumerate(product_depth)

#         local subgroup_basis
#         subgroup_basis = construct_subgroup_basis(lie_basis, product_depth)
#         residuals[i,j], _ = check_target(target, subgroup_basis)

#     end
# end
# plot()
# for (idx, n_qubits) in enumerate(qubit_lst)
#     plot!(product_depth, residuals[idx, :],
#           label="n qubits = $n_qubits",
#           marker=:circle,
#           yscale=:log10,
#           grid = true        
#           )
# end
# xlabel!("Product depth")
# ylabel!("Residual norm")
# # savefig("Res norm")


