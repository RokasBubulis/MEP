include("generators.jl")
include("time_optimal_solver.jl")

# positions = [0 0; 1 0]
# #positions = [0 0; 1 0; 0.5 sqrt(3)/2]
# #positions = [0 0; 1 0; 0 1; 1 1]

# n_qubits = size(positions, 1)
# target =  construct_target_3levels(n_qubits)

# println("PXP model")
# gens = construct_Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis_general(gens)
# println(check_if_implementable(lie_basis, target; print_output = true))
# obtain_target_kak_decomposition_single_control(target, gens)
# println("---")

# println("General 2 controls") 
# target = construct_target_2levels(n_qubits)
# drift = construct_rydberg_drift(positions)
# controls = construct_global_controls(n_qubits)
# #controls = construct_local_controls(n_qubits)
# gens = [drift, controls...]
# lie_basis = construct_lie_basis_general(gens)
# println(check_if_implementable(lie_basis, target; print_output = true))
# obtain_target_kak_decomposition(target, drift, controls)

n_qubits = 2
target = construct_target_3levels(n_qubits)
gens = construct_Ryd_generators(n_qubits)
M0 = kron(id, id)

compute_optimal_time(target, gens, M0; tmax = 5)
