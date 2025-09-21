using Plots, BenchmarkTools, PrettyTables
include("generators.jl")
include("lie_algebra.jl")
include("decomposition.jl")

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

n_qubits = 2
observable = operator(XopRyd([1,2]), n_qubits)
generators = construct_Ryd_generators(n_qubits)
input_matrix = construct_input_matrix(n_qubits)

# n_qubits = 1
# observable = operator(Xop([1]), n_qubits)
# x = sparse(ComplexF64[0 1.0; 1.0 0])
# z = sparse(ComplexF64[1.0 0; 0 -1.0])
# generators = [x,z]
# v = sparsevec([1.0 1.0])/√2
# input_matrix = v*v'

lie_basis = construct_lie_basis(generators, commutation_depth)
adjoint_map = construct_adjoint_representations(lie_basis, generators)

H = generators[1] + generators[2]
H_adjoint = [tr(H' * element) for element in im*lie_basis]

function build_h(H_adjoint, lie_basis)
    h = sum(H_adjoint[i] * im * lie_basis[i] for i in eachindex(H_adjoint))
    return h
end

h = build_h(H_adjoint, lie_basis)
pretty_table(Matrix(H-h))
gram = [tr(B' * C) for B in lie_basis, C in lie_basis]
pretty_table(gram)

# @btime standard_expectation_value(observable, input_matrix, theta, generators)
# @btime gsim_expectation_value(observable, input_matrix, theta, lie_basis, structure_tensor)

# println("Lie basis")
# [pretty_table(a) for a in lie_basis]
# for (idx, O) in enumerate(im*lie_basis)
#     println("Decomposition of element $idx of hermitian basis")
#     pretty_table(O)
#     coeffs = decompose(O, basis_ops)
#     for (name, c) in coeffs
#         println("$name : $c")
#     end
#     println("---")
# end
# println("Decomposition of input density matrix")
# coeffs_input = decompose(convert(SparseMatrixCSC{ComplexF64, Int64}, input_matrix), basis_ops)
# for (name, c) in coeffs_input
#     println("$name : $c")
# end

println("<O> using gsim approach: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map))")
println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")



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