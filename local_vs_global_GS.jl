using BenchmarkTools
include("generators.jl")
include("lie_algebra.jl")

generators = nothing
observable = nothing 
input_matrix = nothing
lie_basis = nothing
adjoint_map = nothing
lie_basis1 = nothing
adjoint_map1 = nothing

for n_qubits in 2:6
    println("Number of qubits: $n_qubits")
    global generators = construct_Ryd_generators(n_qubits)
    global observable = generators[1]
    global input_matrix = construct_input_matrix(n_qubits)

    global lie_basis = construct_lie_basis(generators, commutation_depth)
    global adjoint_map = construct_adjoint_representations(lie_basis, generators)
    println("Global GS. Lie basis dim: $(length(lie_basis))")
    global lie_basis1 = construct_lie_basis_fast(generators, commutation_depth)
    global adjoint_map1 = construct_adjoint_representations(lie_basis1, generators)
    println("Local GS. Lie basis dim: $(length(lie_basis1))")
    println("---")
    println("Benchmarks for Lie basis construction")
    println("global GS:")
    @btime construct_lie_basis(generators, commutation_depth)
    println("local GS:")
    @btime construct_lie_basis_fast(generators, commutation_depth)
    println("---")
    println("Observable expectation values")
    println("Standard <O>: $(standard_expectation_value(observable, input_matrix, theta, generators))")
    println("Lie <O>: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1))")
    println("---")
    println("Simulation benchmarks")
    println("Standard:")
    @btime standard_expectation_value(observable, input_matrix, theta, generators)
    println("Lie:")
    @btime gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1)
    println("---")
    println("---")
end