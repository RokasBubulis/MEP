using Plots
include("generators.jl")
include("lie_algebra.jl")

n_qubits = 2
commutation_depth = 10

generators = construct_Ryd_generators(n_qubits)
lie_basis = construct_lie_basis(generators, commutation_depth)
println("Number of Lie algebra basis elements: $(length(lie_basis))")
#println(Matrix(lie_basis[1]))
repr_elements = construct_repr_elements(lie_basis)
observable = operator(XopRyd([2]), n_qubits)
adjoint_observable = transform_observable_adjoint(observable, lie_basis)
# println(adjoint_observable)

input = construct_input_matrix(n_qubits)
println(observable_expectation(observable, lie_basis, input, [1.0, 0.5]))


# commutation_depth = 10
# n_qubits_list = 2:8
# num_basis_sparse = Int[]
# num_basis_dense = Int[]
# num_basis_ryd = Int[]

# for n in n_qubits_list
#     generators_sparse = construct_sparse_generators(n)
#     lie_basis_sparse = construct_lie_basis(generators_sparse, commutation_depth)
#     l = length(lie_basis_sparse)
#     push!(num_basis_sparse, l)
#     println("Sparse $n : $l")
    
#     generators_dense = construct_dense_generators(n)
#     lie_basis_dense = construct_lie_basis(generators_dense, commutation_depth)
#     l = length(lie_basis_dense)
#     push!(num_basis_dense, l)
#     println("Dense $n : $l")

#     generators_ryd = construct_Ryd_generators(n)
#     lie_basis_ryd = construct_lie_basis(generators_ryd, commutation_depth)
#     l = length(lie_basis_ryd)
#     push!(num_basis_ryd, l)
#     println("Ryd $n : $l")
# end

# # Plot results
# p = plot(n_qubits_list, num_basis_sparse, yscale=:log10, label="Sparse")
# plot!(n_qubits_list, num_basis_dense, label="Dense")
# plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

# xlabel!("Number of Qubits")
# ylabel!("Number of Lie Algebra Basis Elements")
# title!("Commutator depth: $commutation_depth")
# savefig(p, "Lie_algebra_size_depth_$(commutation_depth)_trial.png")