using Plots
include("generators.jl")
include("lie_algebra.jl")


commutation_depth = 10
n_qubits_list = 2:8
num_basis_sparse = Int[]
num_basis_dense = Int[]
num_basis_ryd = Int[]

for n in n_qubits_list
    generators_sparse = construct_sparse_generators(n)
    lie_basis_sparse = construct_lie_basis_fast(generators_sparse, commutation_depth)
    l = length(lie_basis_sparse)
    push!(num_basis_sparse, l)
    println("Sparse $n : $l")
    
    generators_dense = construct_dense_generators(n)
    lie_basis_dense = construct_lie_basis_fast(generators_dense, commutation_depth)
    l = length(lie_basis_dense)
    push!(num_basis_dense, l)
    println("Dense $n : $l")

    generators_ryd = construct_Ryd_generators(n)
    lie_basis_ryd = construct_lie_basis_fast(generators_ryd, commutation_depth)
    l = length(lie_basis_ryd)
    push!(num_basis_ryd, l)
    println("Ryd $n : $l")
end

# Plot results
p = plot(n_qubits_list, num_basis_sparse, yscale=:log10, label="Sparse")
plot!(n_qubits_list, num_basis_dense, label="Dense")
plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

xlabel!("Number of Qubits")
ylabel!("Number of Lie Algebra Basis Elements")
title!("Commutator depth: $commutation_depth")
savefig(p, "Lie_algebra_size_depth_$(commutation_depth)_log.png")

p1 = plot(n_qubits_list, num_basis_sparse, label="Sparse")
plot!(n_qubits_list, num_basis_dense, label="Dense")
plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

xlabel!("Number of Qubits")
ylabel!("Number of Lie Algebra Basis Elements")
title!("Commutator depth: $commutation_depth")
savefig(p1, "Lie_algebra_size_depth_$(commutation_depth).png")