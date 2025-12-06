using SparseArrays, LinearAlgebra

float_type = Complex{Float64}
id = Matrix(float_type[1 0 0; 0 1 0; 0 0 1])

abstract type PauliOp end
struct Xop<: PauliOp
    sites::Vector{Int}
end
struct Zop<: PauliOp
    sites::Vector{Int}
end
struct Yop<: PauliOp
    sites::Vector{Int}
end

struct Qop<: PauliOp
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

operator_matrix(::Xop) = sparse(float_type[0 1; 1 0])
operator_matrix(::Zop) = sparse(float_type[1 0; 0 -1])
operator_matrix(::Yop) = sparse(float_type[0 -im; im 0])
operator_matrix(::Qop) = sparse(float_type[1 0; 0 0])
operator_matrix(::XopRyd) = sparse(float_type[1 0 0; 0 0 1; 0 1 0])
operator_matrix(::ZopRyd) = sparse(float_type[1 0 0; 0 1 0; 0 0 -1])
operator_matrix(::QopRyd) = sparse(float_type[1 0 0; 0 1 0; 0 0 0])

function n_levels(op::PauliOp)
    return 2
end
function n_levels(op::RydbergOp)
    return 3
end

function operator(op::Union{PauliOp, RydbergOp}, n_qubits::Int)
    n_lev = n_levels(op)
    identity_matrix = sparse(Matrix{float_type}(I, n_lev, n_lev))
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
    A = spzeros(float_type, 2^n_qubits, 2^n_qubits)
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
    A = spzeros(float_type, 3^n_qubits, 3^n_qubits)
    for i in 1:n_qubits
        Qnot = sparse(Matrix{float_type}(I, 3^n_qubits, 3^n_qubits))
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

function construct_Ryd_generators_2levels(n_qubits::Int)
    A = spzeros(float_type, 2^n_qubits, 2^n_qubits)
    for i in 1:n_qubits
        Qnot = sparse(Matrix{float_type}(I, 2^n_qubits, 2^n_qubits))
        for j in 1:n_qubits
            if j != i 
                Qnot *= operator(Qop([j]), n_qubits)
            end
        end
        A += operator(Xop([i]), n_qubits) * Qnot
    end
    B = sum(operator(Zop([i]), n_qubits) for i in 1:n_qubits)
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

function Qnot(site::Int, n_qubits::Int)
    @assert n_qubits >= 2 "Number of qubits must be at least 2"
    Qnot = sparse(Matrix{float_type}(I, 3^n_qubits, 3^n_qubits))
    for j in 1:n_qubits
        if j != site
            Qnot *= operator(QopRyd([j]), n_qubits)
        end
    end
    return Qnot
end

function construct_coupled_spin_generators(n_qubits::Int)
    @assert n_qubits >=2 "Number of qubits must be at least 2"
    H_d = 1/2 * operator(Zop([1, 2]), n_qubits)
    H1 = operator(Xop([1]), n_qubits)
    H2 = operator(Yop([1]), n_qubits)
    H3 = operator(Xop([2]), n_qubits)
    H4 = operator(Yop([2]), n_qubits)
    return [H_d, H1, H2, H3, H4]
end

N_LEVELS = 2
function construct_rydberg_drift(positions::AbstractMatrix{<:Real}; C=1, p=6, n_levels = N_LEVELS)
    """
    positions: row per atom
    """
    @assert length(unique(eachrow(positions))) == size(positions, 1) "Positions must be unique"
    N = size(positions, 1)
    dim = n_levels^N
    H_drift = spzeros(float_type, dim, dim)
    V = zeros(Float64, dim, dim)

    for m in 1:N-1
        for n in m+1:N
            r = norm(positions[m,:] - positions[n,:])
            V[m,n] = C / r^p
            V[n,m] = V[m,n]
        end
    end

    for state in 0:dim - 1
        bits = digits(state; base=n_levels, pad=N)
        for m in 1:N-1
            for n in m+1:N
                if (bits[m] == n_levels - 1) && (bits[n] == n_levels - 1)
                    H_drift[state+1, state+1] += V[m,n]
                end
            end
        end
    end
    return H_drift
end

function construct_global_controls(n_qubits; n_levels = N_LEVELS)
    dim = n_levels ^ n_qubits
    X = spzeros(float_type, dim, dim)
    Z = spzeros(float_type, dim, dim)
    for i in 1:n_qubits
        if n_levels == 2
            X += operator(Xop([i]), n_qubits)
            Z += operator(Zop([i]), n_qubits)
        elseif n_levels == 3
            X += operator(XopRyd([i]), n_qubits)
            Z += operator(ZopRyd([i]), n_qubits)
        end
    end
    return [X, Z]
end

function construct_local_controls(n_qubits; n_levels = N_LEVELS)
    x_lst = SparseMatrixCSC{float_type, Int}[]
    z_lst = SparseMatrixCSC{float_type, Int}[]
    for i in 1:n_qubits
        if n_levels == 2
            push!(x_lst, operator(Xop([i]), n_qubits))
            push!(z_lst, operator(Zop([i]), n_qubits))
        elseif n_levels == 3
            push!(x_lst, operator(XopRyd([i]), n_qubits))
            push!(z_lst, operator(ZopRyd([i]), n_qubits))  
        end
    end
    return vcat(x_lst, z_lst)
end