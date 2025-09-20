using Plots, BenchmarkTools, PrettyTables
include("generators.jl")
include("lie_algebra.jl")

function standard_expectation_value(
    observable::SparseMatrixCSC{ComplexF64, Int}, 
    input_matrix::SparseMatrixCSC{Float64, Int}, 
    theta::Vector{Float64}, 
    generators::Vector{SparseMatrixCSC{ComplexF64, Int}})

    unitary = exp(-im*theta[1]*Matrix(generators[1])) * exp(-im*theta[2]*Matrix(generators[2]))
    expectation_value = tr(observable * unitary * input_matrix * unitary')
    return expectation_value
end 

commutation_depth = 10
theta = [1.0, 0.7]

# n_qubits = 2
# observable = operator(Xop([1, 2]), n_qubits)
# # generators = construct_Ryd_generators(n_qubits)
# generators = construct_dense_generators(n_qubits)
# generators /= norm(generators)
# v = sparsevec([0 1.0 1.0 1.0])/√3
# input_matrix = v*v'#input_matrix = construct_input_matrix(n_qubits)
# println("input matrix norm: $(norm(input_matrix))")

n_qubits = 1
observable = operator(Xop([1]), n_qubits)
observable /= norm(observable)
x = sparse(ComplexF64[0 1.0; 1.0 0])
z = sparse(ComplexF64[1.0 0; 0 -1.0])
x = x / norm(x)
z = z / norm(z)
generators = [x,z]
v = sparsevec([1.0 1.0])/√2
input_matrix = v*v'

lie_basis = construct_lie_basis(generators, commutation_depth)
adjoint_map = construct_adjoint_representations(lie_basis, generators)
# @btime standard_expectation_value(observable, input_matrix, theta, generators)
# @btime gsim_expectation_value(observable, input_matrix, theta, lie_basis, structure_tensor)

println("Lie basis")
[pretty_table(a) for a in lie_basis]
println("<O> using gsim approach: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map))")
println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")

# generators = construct_Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis(generators, commutation_depth)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")
# #println(Matrix(lie_basis[1]))
# repr_elements = construct_repr_elements(lie_basis)
# observable = operator(XopRyd([2]), n_qubits)
# adjoint_observable = transform_observable_adjoint(observable, lie_basis)
# # println(adjoint_observable)

# input = construct_input_matrix(n_qubits)
# println(gsim_expectation_value(observable, lie_basis, input, [1.0, 0.5]))


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