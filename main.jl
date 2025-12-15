include("generators.jl")
include("lie_algebra.jl")
include("time_optimal_solver_constrained.jl")

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

# n_qubits = 2
# target = construct_target_3levels(n_qubits)
# gens = construct_Ryd_generators(n_qubits)
# compute_optimal_time(gens, target)


target = operator(YopRyd([1]), 1)
compute_optimal_time(target)

# X = operator(XopRyd([1]), 1)
# Z = operator(ZopRyd([1]), 1)
# gens = [Z, X]
# lie_basis = construct_lie_basis_general(gens)
# @assert check_if_implementable(lie_basis, target) "Target is not implementable"
# p_basis = lie_basis[2:end]
# M1 = build_M(rand(length(p_basis)) .* 2 .- 1, p_basis)
# M2 = build_M(rand(length(p_basis)) .* 2 .- 1, p_basis)
# M3 = build_M(rand(length(p_basis)) .* 2 .- 1, p_basis)
# params = ComponentArray(H0 = -im*X, l = -im*diag(Z), tol = 1e-3, alpha_memory = Ref(1.0)) 
# @show display(M1), display(M2), display(M3)
# H1 = obtain_H_opt(M1, params)
# H2 = obtain_H_opt(M2, params)
# H3 = obtain_H_opt(M3, params)
# @show display(H1), display(H2), display(H3)


