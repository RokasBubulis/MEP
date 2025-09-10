using SparseArrays, LinearAlgebra, Plots

abstract type PauliOp end
struct Xop<: PauliOp
    sites::Vector{Int}
end
struct Zop<: PauliOp
    sites::Vector{Int}
end

abstract type RydbergOp end
struct XopRyd<: RydbergOp
    sites::Vector{Int}
end
struct ZopRyd<: RydbergOp
    sites::Vector{Int}
end
struct QopRyd<: RydbergOp
    sites::Vector{Int}
end

operator_matrix(::Xop) = sparse(ComplexF64[0 1; 1 0])
operator_matrix(::Zop) = sparse(ComplexF64[1 0; 0 -1])
operator_matrix(::XopRyd) = sparse(ComplexF64[1 0 0; 0 0 1; 0 1 0])
operator_matrix(::ZopRyd) = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 -1])
operator_matrix(::QopRyd) = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 0])

function n_levels(op::PauliOp)
    return 2
end
function n_levels(op::RydbergOp)
    return 3
end

function operator(op::Union{PauliOp, RydbergOp}, n_qubits::Int)
    n_lev = n_levels(op)
    identity_matrix = sparse(Matrix{ComplexF64}(I, n_lev, n_lev))
    ops = [identity_matrix for _ in 1:n_qubits]
    mat = operator_matrix(op)
    for s in op.sites
        ops[s] = mat
    end
    result = ops[1]
    for i in 2:n_qubits
        result = kron(result, ops[i])
    end
    return result
end

# function operator(op::PauliOp, n_qubits::Int)
#     ops = [I2 for _ in 1:n_qubits]
#     if op isa Xop
#         for s in op.sites
#             ops[s] = X
#         end
#     elseif op isa Zop
#         for s in op.sites
#             ops[s] = Z
#         end
#     elseif op isa XopRyd
#         for s in op.sites
#             ops[s] = ZRyd
#         end
#     else
#         error("Unknown operator type")
#     end
#     matrix = ops[1]
#     for i in 2:n_qubits
#         matrix = kron(matrix, ops[i])
#     end
#     # matrix = ⊗(ops...)
#     return matrix
# end

function sparse_generators(n_qubits::Int)
    A = sum(operator(Zop([i]), n_qubits) * operator(Zop([i+1]), n_qubits) for i in 1:n_qubits-1)
    B = sum(operator(Xop([i]), n_qubits) for i in 1:n_qubits)
    return (A, B)
end

function dense_generators(n_qubits::Int)
    A = spzeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:n_qubits-1
        for j in i+1:n_qubits
            Jij = 1 / abs(i - j)
            A += Jij * operator(Zop([i]), n_qubits) * operator(Zop([j]), n_qubits)
        end
    end
    B = sum(operator(Xop([i]), n_qubits) for i in 1:n_qubits)
    return (A, B)
end

function Ryd_generators(n_qubits::Int)
    A = spzeros(ComplexF64, 3^n_qubits, 3^n_qubits)
    for i in 1:n_qubits
        Qnot = sparse(Matrix{ComplexF64}(I, 3^n_qubits, 3^n_qubits))
        for j in 1:n_qubits
            if j != i 
                Qnot *= operator(QopRyd([j]), n_qubits)
            end
        end
        A += operator(XopRyd([i]), n_qubits) * Qnot
    end
    B = sum(operator(ZopRyd([i]), n_qubits) for i in 1:n_qubits)
    return (A, B)
end

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

# function construct_lie_basis(generators::Tuple{Vararg{SparseMatrixCSC{ComplexF64, Int}}}, depth::Int)
#     gen1, gen2 = generators
#     basis_elements = SparseMatrixCSC{ComplexF64,Int}[gen1, gen2]
#     bracket_rhs = im*gen2
#     for k in 1:depth
#         bracket = br(im*gen1, bracket_rhs)
#         if is_independent(basis_elements, bracket)
#             println(k)
#             push!(basis_elements, bracket)
#         end
#         bracket_rhs = bracket
#     end
#     return basis_elements
# end

# function construct_lie_basis(generators::Tuple{Vararg{SparseMatrixCSC{ComplexF64, Int}}}, depth::Int)
#     gen1, gen2 = generators
#     basis_elements = SparseMatrixCSC{ComplexF64,Int}[gen1, gen2]
#     bracket = br(gen1, gen2)
#     if is_independent(basis_elements, bracket)
#         push!(basis_elements, bracket)
#     else
#         return basis_elements
#     end
#     all_elements = copy(basis_elements)
#     for _ in 2:depth
#         len = length(basis_elements)
#         a = 0
#         for operator in basis_elements[3+a:len]
#             bracket1 = br(gen1, operator)
#             if is_independent(basis_elements, bracket1)
#                 push!(basis_elements, bracket1)
#                 a += 1
#             end
#             bracket2 = br(gen2, operator)
#             if is_independent(basis_elements, bracket2)
#                 push!(basis_elements, bracket2)
#                 a += 1
#             end
#         end
#     end

#     return basis_elements
# end

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


commutation_depth = 10
n_qubits_list = 2:6 
num_basis_sparse = Int[]
num_basis_dense = Int[]
num_basis_ryd = Int[]

for n in n_qubits_list
    generators_sparse = sparse_generators(n)
    lie_basis_sparse = construct_lie_basis(generators_sparse, commutation_depth)
    l = length(lie_basis_sparse)
    push!(num_basis_sparse, l)
    println("Sparse $n : $l")
    
    # generators_dense = dense_generators(n)
    # lie_basis_dense = construct_lie_basis(generators_dense, commutation_depth)
    # push!(num_basis_dense, length(lie_basis_dense))

    generators_ryd = Ryd_generators(n)
    lie_basis_ryd = construct_lie_basis(generators_ryd, commutation_depth)
    l = length(lie_basis_ryd)
    push!(num_basis_ryd, l)
    println("Ryd $n : $l")
end

# Plot results
plot(n_qubits_list, num_basis_sparse, label="Sparse")
# plot!(n_qubits_list, num_basis_dense, label="Dense")
plot!(n_qubits_list, num_basis_ryd, label="Rydberg")

xlabel!("Number of Qubits")
ylabel!("Number of Lie Algebra Basis Elements")
title!("Lie Basis Size vs Number of Qubits")
