using Plots, BenchmarkTools, PrettyTables, SparseArrays
include("generators.jl")
include("lie_algebra.jl")
include("decomposition.jl")

function standard_expectation_value(
    observable::SparseMatrixCSC{ComplexF64, Int}, 
    input_matrix::SparseMatrixCSC{Float64, Int}, 
    theta::Vector{Float64}, 
    generators::Vector{SparseMatrixCSC{ComplexF64, Int}})

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

commutation_depth = 10
theta = [0.7, 0.6]

n_qubits = 8
#observable = operator(XopRyd([1,2]), n_qubits) + operator(XopRyd([2]), n_qubits)
generators = construct_Ryd_generators(n_qubits)
observable = generators[1]
input_matrix = construct_input_matrix(n_qubits)

# lie_basis = construct_lie_basis(generators, commutation_depth)
# adjoint_map = construct_adjoint_representations(lie_basis, generators)
# println("lie basis 1: $(length(lie_basis))")
# @btime construct_lie_basis(generators, commutation_depth)

lie_basis1 = construct_lie_basis_fast(generators, commutation_depth)
adjoint_map1 = construct_adjoint_representations(lie_basis1, generators)
println("lie basis 2: $(length(lie_basis1))")
@btime construct_lie_basis_fast(generators, commutation_depth)

println("<O> using standard approach: $(standard_expectation_value(observable, input_matrix, theta, generators))")
println("<O> using gsim approach 2: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1))")

# generators = nothing
# observable = nothing 
# input_matrix = nothing
# lie_basis = nothing
# adjoint_map = nothing
# lie_basis1 = nothing
# adjoint_map1 = nothing
# for n_qubits in 2:7
#     println("Number of qubits: $n_qubits")
#     #observable = operator(XopRyd([1,2]), n_qubits) + operator(XopRyd([2]), n_qubits)
#     global generators = construct_Ryd_generators(n_qubits)
#     global observable = generators[1]
#     global input_matrix = construct_input_matrix(n_qubits)

#     global lie_basis = construct_lie_basis(generators, commutation_depth)
#     global adjoint_map = construct_adjoint_representations(lie_basis, generators)
#     println("Global GS. Lie basis dim: $(length(lie_basis))")
#     global lie_basis1 = construct_lie_basis_fast(generators, commutation_depth)
#     global adjoint_map1 = construct_adjoint_representations(lie_basis1, generators)
#     println("Local GS. Lie basis dim: $(length(lie_basis1))")
#     println("---")
#     println("Benchmarks for Lie basis construction")
#     println("global GS:")
#     @btime construct_lie_basis(generators, commutation_depth)
#     println("local GS:")
#     @btime construct_lie_basis_fast(generators, commutation_depth)
#     println("---")
#     println("Observable expectation values")
#     println("Standard <O>: $(standard_expectation_value(observable, input_matrix, theta, generators))")
#     println("Global Lie <O>: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis, adjoint_map))")
#     println("Local Lie <O>: $(gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1))")
#     println("---")
#     println("Simulation benchmarks")
#     println("Standard:")
#     @btime standard_expectation_value(observable, input_matrix, theta, generators)
#     println("Lie:")
#     @btime gsim_expectation_value(observable, input_matrix, theta, lie_basis1, adjoint_map1)
#     println("---")
#     println("---")
# end

# commutation_depth = 10
# n_qubits_list = 2:5
# num_basis_sparse = Int[]
# num_basis_dense = Int[]
# num_basis_ryd = Int[]

# for n in n_qubits_list
#     generators_sparse = construct_sparse_generators(n)
#     lie_basis_sparse = construct_lie_basis(generators_sparse, commutation_depth)
#     l = length(lie_basis_sparse)
#     push!(num_basis_sparse, l)
#     println("Sparse $n : $l")
    
#     generators_dense = construct_dense_generators(n)
#     lie_basis_dense = construct_lie_basis(generators_dense, commutation_depth)
#     l = length(lie_basis_dense)
#     push!(num_basis_dense, l)
#     println("Dense $n : $l")

#     generators_ryd = construct_Ryd_generators(n)
#     lie_basis_ryd = construct_lie_basis(generators_ryd, commutation_depth)
#     l = length(lie_basis_ryd)
#     push!(num_basis_ryd, l)
#     println("Ryd $n : $l")
# end

# # Plot results
# p = plot(n_qubits_list, num_basis_sparse, yscale=:log10, label="Sparse")
# plot!(n_qubits_list, num_basis_dense, label="Dense")
# plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

# xlabel!("Number of Qubits")
# ylabel!("Number of Lie Algebra Basis Elements")
# title!("Commutator depth: $commutation_depth")
# savefig(p, "Lie_algebra_size_depth_$(commutation_depth)_trial.png")