using BenchmarkTools
include("generators.jl")
include("lie_algebra.jl")

commutation_depth = 10
theta = [0.7, 0.6]

n_qubits = 8
#observable = operator(XopRyd([1,2]), n_qubits) + operator(XopRyd([2]), n_qubits)
generators = construct_Ryd_generators(n_qubits)
observable = generators[1]
input_matrix = construct_input_matrix(n_qubits)

# lie_basis = construct_lie_basis(generators, commutation_depth)
# adjoint_map = construct_adjoint_representations(lie_basis, generators)
# println("lie basis 1: $(length(lie_basis))")
# @btime construct_lie_basis(generators, commutation_depth)

lie_basis1 = construct_lie_basis_fast(generators, commutation_depth)
adjoint_map1 = construct_adjoint_representations(lie_basis1, generators)
println("lie basis 2: $(length(lie_basis1))")
@btime construct_lie_basis_fast(generators, commutation_depth)

println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")
println("<O> using gsim approach 2: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1))")