include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")
include("time_optimal_solution.jl")
using BenchmarkTools

#positions = [0 0; 1 0]
positions = [0 0; 1 0; 0.5 sqrt(3)/2]
#positions = [0 0; 1 0; 0 1; 1 1]

n_qubits = size(positions, 1)
drift = construct_rydberg_drift(positions)

controls = construct_global_controls(n_qubits)
#controls = construct_local_controls(n_qubits)

#target =  construct_target_2levels(n_qubits)
target = exp(im*pi/4)*construct_target_2levels(n_qubits)

gens = copy(controls)
push!(gens, drift)
lie_basis = construct_lie_basis_general(gens, 10)
println(check_if_implementable(lie_basis, target, 10; print_output = true))
obtain_target_kak_decomposition(target, drift, controls)


# n_qubits = 2
# gens = construct_coupled_spin_generators(n_qubits)
# target = exp(im*pi/4)*construct_target_2levels(n_qubits)
# lie_basis = construct_lie_basis_general(gens, 10)
# println(check_if_implementable(lie_basis, target, 10; print_output = true))
# drift, controls = gens[1], gens[2:end]
# obtain_target_kak_decomposition(target, drift, controls)


# notes
# lobal controls: works for all with standard target
# global controls: 2 qubits works with standard, 3 qubits works with global phase applied, though cartan error is thrown??