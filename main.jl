using BenchmarkTools
include("generators.jl")
include("lie_algebra.jl")

function check_observable(observable::SparseMatrixCSC{ComplexF64, Int}, lie_basis::Vector{SparseMatrixCSC{ComplexF64, Int}})
    v_obs = Vector{eltype(observable)}(undef, length(lie_basis))
    temp_element = similar(observable)
    for (i, element) in enumerate(lie_basis)
        mul!(temp_element, element', observable)
        v_obs[i] = im*tr(temp_element)
    end
    reconstructed_obs = similar(observable)
    for i in eachindex(v_obs)
        reconstructed_obs += v_obs[i] * lie_basis[i]/im
    end
    return norm(reconstructed_obs - observable)
end

commutation_depth = 10
theta = [0.7, 0.6]
n_qubits = 8
alpha = 0.1

# multiple_excitations = alpha * operator(XopRyd([1]), n_qubits) * operator(QopRyd([2]), n_qubits) * operator(XopRyd([3]), n_qubits)
# gen1 = sum([operator(XopRyd([i]), n_qubits) * Qnot(i, n_qubits) for i in 1:n_qubits]) + multiple_excitations
# gen2 = operator(ZopRyd([1,2]), n_qubits) + operator(ZopRyd([2,3]), n_qubits)
# generators = [gen1, gen2]

# lie_basis = construct_lie_basis_fast(generators, commutation_depth)
# println("Dim lie basis: $(length(lie_basis))")
# observable = gen2

# println(check_observable(observable, lie_basis))


#observable = operator(XopRyd([1,2]), n_qubits) + operator(XopRyd([2]), n_qubits)
generators = construct_Ryd_generators(n_qubits)
observable = generators[1]
input_matrix = construct_input_matrix(n_qubits)

lie_basis = construct_lie_basis_fast(generators, commutation_depth)
adjoint_map = construct_adjoint_representations(lie_basis, generators)
println("lie basis: $(length(lie_basis))")

# @btime standard_expectation_value(observable, input_matrix, theta, generators)
# println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")
@btime gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map)
println("<O> using gsim approach: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map))")