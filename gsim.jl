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

generators = construct_Ryd_generators(n_qubits)
observable = generators[1]
input_matrix = construct_input_matrix(n_qubits)

lie_basis = construct_system_subgroup_basis(generators, commutation_depth)
adjoint_map = construct_adjoint_representations(lie_basis, generators)
println("lie basis: $(length(lie_basis))")

# @btime standard_expectation_value(observable, input_matrix, theta, generators)
# println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")
@btime gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map)
println("<O> using gsim approach: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map))")