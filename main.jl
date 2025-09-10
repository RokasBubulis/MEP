using Plots
include("generators.jl")

br(A, B) = A * B - B * A

function is_independent(matrix_list::Vector{<:SparseMatrixCSC{ComplexF64, Int}}, candidate_matrix::SparseMatrixCSC{ComplexF64, Int}; tol=1e-10)
    if isempty(matrix_list)
        return true
    end
    M = hcat([vec(C) for C in matrix_list]...)
    d = Array(vec(candidate_matrix))
    try
        x = M \ d
        residual = norm(M * x - d)
        return residual ≥ tol
    catch e
        if isa(e, SingularException)
            return false
        else
            rethrow(e)
        end
    end
end

function construct_lie_basis(generators::Tuple{Vararg{SparseMatrixCSC{ComplexF64, Int}}}, depth::Int)
    gen1, gen2 = generators
    bracket = br(gen1, gen2)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[gen1, gen2]
    if is_independent(basis_elements, bracket)
        push!(basis_elements, bracket)
    end
    new_elements = SparseMatrixCSC{ComplexF64,Int}[bracket]
    depth -= 1
    while depth > 0
        next_layer = SparseMatrixCSC{ComplexF64,Int}[]
        for element in new_elements
            for gen in generators
                bracket = br(gen, element)
                push!(next_layer, bracket)
                if is_independent(basis_elements, bracket)
                    push!(basis_elements, bracket)
                end
            end
        end
        new_elements = next_layer
        depth -= 1 
    end 
    return basis_elements
end
function construct_repr_elements(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
    n = length(lie_basis)
    repr_elements = zeros(ComplexF64, n, n, n)
    for alpha in 1:n
        for beta in 1:n
            for gamma in 1:n
                repr_elements[alpha,beta,gamma] = tr(br(im*lie_basis[beta], br(lie_basis[gamma], im*lie_basis[alpha])))
            end
        end
    end
    return repr_elements
end

function transform_observable_adjoint(observable::SparseMatrixCSC{ComplexF64, Int}, lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
    coeffs = [tr(observable * im*element) for element in lie_basis]
    return coeffs
end

##

# n_qubits = 3
# commutation_depth = 10

# generators_sparse = sparse_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_sparse, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(Zop([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


# generators_dense = dense_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_dense, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(Zop([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


# generators_ryd = Ryd_generators(n_qubits)
# lie_basis = construct_lie_basis(generators_ryd, commutation_depth)
# # repr_elements = construct_repr_elements(lie_basis)
# # observable = operator(ZopRyd([1,2]), n_qubits)
# println("Number of Lie algebra basis elements: $(length(lie_basis))")


commutation_depth = 8
n_qubits_list = 2:6 
num_basis_sparse = Int[]
num_basis_dense = Int[]
num_basis_ryd = Int[]

for n in n_qubits_list
    generators_sparse = sparse_generators(n)
    lie_basis_sparse = construct_lie_basis(generators_sparse, commutation_depth)
    l = length(lie_basis_sparse)
    push!(num_basis_sparse, l)
    # println("Sparse $n : $l")
    
    generators_dense = dense_generators(n)
    lie_basis_dense = construct_lie_basis(generators_dense, commutation_depth)
    push!(num_basis_dense, length(lie_basis_dense))

    generators_ryd = Ryd_generators(n)
    lie_basis_ryd = construct_lie_basis(generators_ryd, commutation_depth)
    l = length(lie_basis_ryd)
    push!(num_basis_ryd, l)
    # println("Ryd $n : $l")
end

# Plot results
p = plot(n_qubits_list, num_basis_sparse, label="Sparse")
plot!(n_qubits_list, num_basis_dense, label="Dense")
plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

xlabel!("Number of Qubits")
ylabel!("Number of Lie Algebra Basis Elements")
title!("Commutator depth: $commutation_depth")
savefig(p, "Lie_algebra_size_depth_$commutation_depth.png")
