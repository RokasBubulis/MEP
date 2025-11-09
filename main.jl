include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")
using BenchmarkTools

commutation_depth = 10
max_product_depth = 15

qubit_lst = 2:8
for n_qubits in qubit_lst
    local target, generators, lie_basis
    println("N qubits: $n_qubits")
    target = construct_target(n_qubits)
    generators = construct_Ryd_generators(n_qubits)
    println("Construct algebra basis time:")
    @btime construct_lie_basis($generators, commutation_depth)
    lie_basis = construct_lie_basis(generators, commutation_depth)
    println("Construct group basis time:")
    @btime check_if_implementable($lie_basis, $target, $max_product_depth)
    println(check_if_implementable(lie_basis, target, max_product_depth))
    println("---")
end
