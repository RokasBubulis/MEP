include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")
include("time_optimal_solution.jl")
using BenchmarkTools

#positions = [0 0; 1 0]
positions = [0 0; 1 0; 0.5 sqrt(3)/2]
# positions = [0 0; 1 0; 0 1; 1 1]

n_qubits = size(positions, 1)
drift = construct_rydberg_drift(positions)
# controls = construct_global_controls(n_qubits)
controls = construct_local_controls(n_qubits)
# controls = construct_Ryd_generators_2levels(n_qubits)
target = construct_target_2levels(n_qubits)

gens = copy(controls)
push!(gens, drift)
lie_basis = construct_lie_basis_general(gens, 10)
println(check_if_implementable(lie_basis, target, 10; print_output = true))
obtain_target_kak_decomposition(target, drift, controls)

# display(operator(Xop([1]), 2) + operator(Xop([2]), 2))
# display(operator(Zop([1]), 2) + operator(Zop([2]), 2))