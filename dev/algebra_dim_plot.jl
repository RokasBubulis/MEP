using Plots
include("../src/generators.jl")
include("../src/lie_algebra.jl")


commutation_depth = 15
n_qubits_list = 2:6
num_basis_sparse = Int[]
num_basis_dense = Int[]
num_basis_ryd = Int[]

for n in n_qubits_list
    generators_sparse = construct_sparse_generators(n)
    lie_basis_sparse = construct_lie_basis_general(generators_sparse; depth=commutation_depth)
    l = length(lie_basis_sparse)
    push!(num_basis_sparse, l)
    println("Sparse $n : $l")
    
    generators_dense = construct_dense_generators(n)
    lie_basis_dense = construct_lie_basis_general(generators_dense; depth=commutation_depth)
    l = length(lie_basis_dense)
    push!(num_basis_dense, l)
    println("Dense $n : $l")

    generators_ryd = construct_Ryd_generators(n)
    lie_basis_ryd = construct_lie_basis_general(generators_ryd; depth=commutation_depth)
    l = length(lie_basis_ryd)
    push!(num_basis_ryd, l)
    println("Ryd $n : $l")
end

# Plot results
p = plot(n_qubits_list, num_basis_sparse,
    yscale=:log10,
    label="Sparse",
    marker=:circle,
    markersize=3,
    grid=true,
    gridlinewidth=0.5,
    gridalpha=0.4,
    minorgrid=true,
    minorgridalpha=0.2
)
plot!(n_qubits_list, num_basis_dense, yscale=:log10, label="Dense", marker=:circle, markersize=3)
plot!(n_qubits_list, num_basis_ryd, yscale=:log10, label="Rydberg", marker=:circle, markersize=3)
su_qubit = [n_qubits^4 - 1 for n_qubits in n_qubits_list]
su_qutrit = [n_qutrits^6 - 1 for n_qutrits in n_qubits_list]
plot!(n_qubits_list, su_qubit, yscale=:log10, label="su(n qubits)", marker=:circle, markersize=2)
plot!(n_qubits_list, su_qutrit, yscale=:log10, label="su(n qutrits)", marker=:circle, markersize=2)

alignments = :bottom
factor_sp = 1
factor_ry = 1
factor_de = 1
for (i, n) in enumerate(n_qubits_list)
    if i == 2
        factor_sp = 0.6
        factor_ry = 1.1
        factor_de = 1.1
    elseif i == 3 || i == 4
        factor_sp = 1.1
        factor_ry = 0.6
        factor_de = 1
    elseif i == 5
        factor_sp = 1
        factor_ry = 1
        factor_de = 0.5
    else
        factor_sp = 1
        factor_ry = 1
        factor_de = 1
    end 
    annotate!(n, num_basis_sparse[i] * factor_sp, text(string(num_basis_sparse[i]), 9, :center, alignments))
    annotate!(n, num_basis_dense[i] *factor_de, text(string(num_basis_dense[i]),  9, :center, alignments))
    annotate!(n, num_basis_ryd[i] * factor_ry  , text(string(num_basis_ryd[i]),    9, :center, alignments))
    annotate!(n, su_qubit[i]        , text(string(su_qubit[i]),         9, :center, alignments))
    annotate!(n, su_qutrit[i]       , text(string(su_qutrit[i]),        9, :center, alignments))
end

xlabel!("Number of particles")
ylabel!("Dimension of Lie algebra")
title!("Max commutator depth: $commutation_depth")
savefig(p, "results/Lie_algebra_size_depth_$(commutation_depth)_log.png")
