using Plots
include("generators.jl")
include("lie_algebra.jl")

# function construct_repr_elements(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
#     n = length(lie_basis)
#     repr_elements = zeros(ComplexF64, n, n, n)
#     for alpha in 1:n
#         for beta in 1:n
#             for gamma in 1:n
#                 repr_elements[alpha,beta,gamma] = tr(br(im*lie_basis[beta], br(lie_basis[gamma], im*lie_basis[alpha])))
#             end
#         end
#     end
#     return repr_elements
# end

# function transform_observable_adjoint(observable::SparseMatrixCSC{ComplexF64, Int}, lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
#     coeffs = [tr(observable * im*element) for element in lie_basis]
#     return coeffs
# end

##

# n_qubits = 3
# commutation_depth = 10
# generators_ryd = Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_ryd, commutation_depth)
# println(length(lie_basis))

# generators_sparse = sparse_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_sparse, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(Zop([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


# generators_dense = dense_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_dense, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(Zop([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


# generators_ryd = Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_ryd, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(ZopRyd([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


commutation_depth = 10
n_qubits_list = 2:7
num_basis_sparse = Int[]
num_basis_dense = Int[]
num_basis_ryd = Int[]

for n in n_qubits_list
    generators_sparse = construct_sparse_generators(n)
    lie_basis_sparse = construct_lie_basis(generators_sparse, commutation_depth)
    l = length(lie_basis_sparse)
    push!(num_basis_sparse, l)
    println("Sparse $n : $l")
    
    # generators_dense = construct_dense_generators(n)
    # lie_basis_dense = construct_lie_basis(generators_dense, commutation_depth)
    # l = length(lie_basis_dense)
    # push!(num_basis_dense, l)
    # println("Dense $n : $l")

    generators_ryd = construct_Ryd_generators(n)
    lie_basis_ryd = construct_lie_basis(generators_ryd, commutation_depth)
    l = length(lie_basis_ryd)
    push!(num_basis_ryd, l)
    println("Ryd $n : $l")
end

# Plot results
p = plot(n_qubits_list, num_basis_sparse, label="Sparse")
# plot!(n_qubits_list, num_basis_dense, label="Dense")
plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

xlabel!("Number of Qubits")
ylabel!("Number of Lie Algebra Basis Elements")
title!("Commutator depth: $commutation_depth")
savefig(p, "Lie_algebra_size_depth_$(commutation_depth)_not_dense.png")