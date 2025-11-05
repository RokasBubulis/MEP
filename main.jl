include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")
using Plots, BenchmarkTools

commutation_depth = 10
max_product_depth = 15

qubit_lst = 2:7
for n_qubits in qubit_lst
    local target, generators, lie_basis
    target = construct_target(n_qubits)
    generators = construct_Ryd_generators(n_qubits)
    lie_basis = construct_lie_basis_fast(generators, commutation_depth)
    println("N qubits: $n_qubits")
    # @btime check_if_implementable(lie_basis, target, max_product_depth)
    println(check_if_implementable(lie_basis, target, max_product_depth))
    println("---")
end


# qubit_lst = 2:5
# product_depth = 1:4
# residuals = zeros(length(qubit_lst), length(product_depth))
# for (i, n_qubits) in enumerate(qubit_lst)

#     println("n qubits: $n_qubits")
#     local generators, target, lie_basis
#     generators = construct_Ryd_generators(n_qubits)
#     target = construct_target(n_qubits)
#     lie_basis = construct_lie_basis_fast(generators, commutation_depth)

#     for (j, product_depth) in enumerate(product_depth)

#         local subgroup_basis
#         subgroup_basis = construct_subgroup_basis(lie_basis, product_depth)
#         residuals[i,j], _ = check_target(target, subgroup_basis)

#     end
# end
# plot()
# for (idx, n_qubits) in enumerate(qubit_lst)
#     plot!(product_depth, residuals[idx, :],
#           label="n qubits = $n_qubits",
#           marker=:circle,
#           yscale=:log10,
#           grid = true        
#           )
# end
# xlabel!("Product depth")
# ylabel!("Residual norm")
# savefig("Res norm")


