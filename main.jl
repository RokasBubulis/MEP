include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")
include("time_optimal_solution.jl")
using BenchmarkTools

positions = [0 0; 1 0]
#positions = [0 0; 1 0; 0.5 sqrt(3)/2]
#positions = [0 0; 1 0; 0 1; 1 1]

n_qubits = size(positions, 1)
target =  construct_target_3levels(n_qubits)

println("PXP model")
gens = construct_Ryd_generators(n_qubits)
lie_basis = construct_lie_basis_general(gens)
println(check_if_implementable(lie_basis, target; print_output = true))
obtain_target_kak_decomposition_single_control(target, gens)
println("---")

println("General 2 controls") 
target = construct_target_2levels(n_qubits)
drift = construct_rydberg_drift(positions)
controls = construct_global_controls(n_qubits)
#controls = construct_local_controls(n_qubits)
gens = [drift, controls...]
lie_basis = construct_lie_basis_general(gens)
println(check_if_implementable(lie_basis, target; print_output = true))
obtain_target_kak_decomposition(target, drift, controls)


# n_qubits = 2
# gens = construct_coupled_spin_generators(n_qubits)
# target = exp(im*pi/4)*construct_target_2levels(n_qubits)
# lie_basis = construct_lie_basis_general(gens, 10)
# println(check_if_implementable(lie_basis, target, 10; print_output = true))
# drift, controls = gens[1], gens[2:end]
# obtain_target_kak_decomposition(target, drift, controls)


# notes
# local controls: works for all with standard target
# global controls: 2 qubits works with standard, 3 qubits works with global phase applied, though cartan error is thrown??
# that is a wrong global phase for 3 qubits if we wanted to make determinant 1
# example of local controls for 2 qubits, only works if adjusting target by exp(im*pi/4)