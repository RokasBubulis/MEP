using SparseArrays, LinearAlgebra

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

function construct_sparse_generators(n_qubits::Int)
    A = sum(operator(Zop([i]), n_qubits) * operator(Zop([i+1]), n_qubits) for i in 1:n_qubits-1)
    B = sum(operator(Xop([i]), n_qubits) for i in 1:n_qubits)
    return [A, B]
end

function construct_dense_generators(n_qubits::Int)
    A = spzeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:n_qubits-1
        for j in i+1:n_qubits
            Jij = 1 / abs(i - j)
            A += Jij * operator(Zop([i]), n_qubits) * operator(Zop([j]), n_qubits)
        end
    end
    B = sum(operator(Xop([i]), n_qubits) for i in 1:n_qubits)
    return [A, B]
end

function construct_Ryd_generators(n_qubits::Int)
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
    return [A, B]
end

function construct_input_matrix(n_qubits::Int)
    v = sparsevec([1, 0, 0])
    result = copy(v)
    if n_qubits >= 3
        for _ in 1:n_qubits-2
            result = kron(result, v)
        end
    end
    result = kron(result, 1/sqrt(2)*sparsevec([1, 1, 0]))
    return result * transpose(result)
end