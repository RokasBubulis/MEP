include("generators.jl")
include("lie_algebra.jl")

using Plots

function check_target(target::SparseMatrixCSC{ComplexF64, Int}, system_basis::Vector{SparseMatrixCSC{ComplexF64, Int}})

    reconstructed_target = spzeros(ComplexF64, size(target))
    for i in eachindex(system_basis)
        coeff = tr(adjoint(system_basis[i]) * target)
        reconstructed_target += coeff * system_basis[i]
    end

    residual = reconstructed_target - target
    return norm(residual), residual
end


function construct_target(n_qubits::Int)
    I3 = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    target = copy(I3)
    for n in 2:n_qubits
        target = kron(target, I3)
    end
    target[3^n_qubits, 3^n_qubits] = -1.0
    return target
end

function construct_system_subgroup_basis(H_lie::SparseMatrixCSC{ComplexF64, Int}, n_terms::Int)
    system_basis = SparseMatrixCSC{ComplexF64,Int}[]
    for n in 0:(n_terms - 1)
        candidate_element = (-im*H_lie)^n / factorial(n)
        try_add_orthonormal!(system_basis, candidate_element)
    end
    return system_basis
end

# generators = construct_Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis(generators, commutation_depth)
# H_lie = im*sum(lie_basis)
# system_basis = construct_system_subgroup_basis(H_lie, lie_power_terms)
# target = construct_target(n_qubits)
# println("Lie basis elements: $(length(lie_basis)), System subgroup elements: $(length(system_basis))")
# residual_norm, residual = check_target(target, system_basis)
# println("Residual norm: $(residual_norm)")

commutation_depth = 10
qubit_lst = 2:6
power_lst = 1:10

residuals = zeros(length(qubit_lst),length(power_lst))

for (i,n_qubits) in enumerate(qubit_lst)
    for (j,lie_power_terms) in enumerate(power_lst)
        local generators, target, lie_basis, H_lie, system_basis

        generators = construct_Ryd_generators(n_qubits)
        target = construct_target(n_qubits)
        lie_basis = construct_lie_basis_fast(generators, commutation_depth)
        H_lie = im*sum(lie_basis)
        system_basis = construct_system_subgroup_basis(H_lie, lie_power_terms)
        residuals[i,j], _ = check_target(target, system_basis)
    end
end

plot()
for (idx, n_qubits) in enumerate(qubit_lst)
    plot!(power_lst, residuals[idx, :],
          label="n_qubits = $n_qubits",
          marker=:circle,
        #   yscale=:log10          
          )
end

xlabel!("Lie power terms")
ylabel!("Residual norm")
# savefig("Res norm log")

