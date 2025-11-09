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

# Default matrix exponentiation approach
function standard_expectation_value(
    observable::SparseMatrixCSC{float_type, Int}, 
    input_matrix::SparseMatrixCSC{Float64, Int}, 
    theta::Vector{Float64}, 
    generators::Vector{SparseMatrixCSC{float_type, Int}})

    tmpH = similar(Matrix(generators[1]))
    mul!(tmpH, Matrix(generators[1]), theta[1])
    BLAS.axpy!(theta[2], Matrix(generators[2]), tmpH) 
    lmul!(-im, tmpH)                           
    unitary = exp(tmpH) 
    tmp1 = similar(unitary)
    tmp2 = similar(unitary)
    tmp3 = similar(unitary)
    mul!(tmp1, observable, unitary)
    mul!(tmp2, tmp1, input_matrix)
    mul!(tmp3, tmp2, unitary')                  
    return tr(tmp3)
end 


function gsim_expectation_value(
        observable::SparseMatrixCSC{float_type, Int},
        input_matrix::SparseMatrixCSC{Float64, Int},
        theta::Vector{Float64},
        lie_basis::Vector{SparseMatrixCSC{float_type,Int}},
        adjoint_map::Vector{Matrix{float_type}})

    # v_in = Vector{eltype(observable)}(undef, length(lie_basis))
    # temp_element = similar(observable) 
    # for (i, element) in enumerate(lie_basis)
    #     mul!(temp_element, element', input_matrix)
    #     v_in[i] = im*tr(temp_element)
    # end
    # v_obs = Vector{eltype(observable)}(undef, length(lie_basis))
    # temp_element = similar(observable)
    # for (i, element) in enumerate(lie_basis)
    #     mul!(temp_element, element', observable)
    #     v_obs[i] = im*tr(temp_element)
    # end

    n = size(lie_basis[1], 1)
    n_basis = length(lie_basis)
    vec_basis = [vec(im * b) for b in lie_basis]
    vec_input = vec(input_matrix)
    vec_obs   = vec(observable)

    v_in  = [dot(b, vec_input) for b in vec_basis]
    v_obs = [dot(b, vec_obs)   for b in vec_basis]

    A, B = adjoint_map[1], adjoint_map[2]
    tmp = similar(A)                      
    tmp .= theta[1] .* A    
    tmp .+= theta[2] .* B   
    tmp .*= -im             
    circuit = exp(tmp)    
    v_out = similar(v_in)
    mul!(v_out, circuit, v_in)
    return dot(v_out, v_obs)
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