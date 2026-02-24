include("generators.jl")
include("lie_algebra.jl")
include("group_implementability.jl")


function construct_H_target(n_qubits::Int, n_levels::Int)
    dim = n_levels^n_qubits
    H = spzeros(ComplexF64, dim, dim)
    H[5, 5] = pi
    H -= spdiagm(0 => fill(tr(H)/dim, dim))
    return H
end

n_qubits = 2
n_levels = 3
control, drift = construct_Ryd_generators(n_qubits)
generators = [control, drift]
target = construct_H_target(n_qubits, n_levels)
display(target)

for order in 1:6
    lie_algebra = construct_lie_basis_general(generators; depth=order)
    println("Order: $order, Algebra dim: $(length(lie_algebra))")
    check_group_implementability(target, lie_algebra)
end
