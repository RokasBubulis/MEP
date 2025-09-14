
br(A, B) = A * B - B * A

function is_independent(matrix_list::Vector{<:SparseMatrixCSC{ComplexF64, Int}}, candidate_matrix::SparseMatrixCSC{ComplexF64, Int}; tol_rel=1e-12)
    if isempty(matrix_list)
        return true
    end
    M = hcat([vec(C) for C in matrix_list]...)
    d = Array(vec(candidate_matrix))
    try
        x = M \ d
        residual = norm(M * x - d)  / norm(d)
        return residual ≥ tol_rel
    catch e
        if isa(e, SingularException)
            return false
        else
            rethrow(e)
        end
    end
end

function construct_lie_basis(generators::Tuple{Vararg{SparseMatrixCSC{ComplexF64, Int}}}, depth::Int)
    gen1, gen2 = im .* generators
    bracket = br(gen1, gen2)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[gen1, gen2]
    if is_independent(basis_elements, bracket)
        push!(basis_elements, bracket)
    end
    new_elements = [bracket] 
    d = 1 # first order nested commutator already included
    while d < depth
        d += 1
        next_layer = SparseMatrixCSC{ComplexF64, Int}[]
        for element in new_elements
            for gen in generators
                    bracket = br(im*gen, element)
                    if is_independent(basis_elements, bracket)
                        push!(next_layer, bracket)
                        push!(basis_elements, bracket)
                    end
            end
        end
        new_elements = next_layer
    end 
    basis_elements = [X / norm(X) for X in basis_elements]
    return basis_elements
end

function construct_repr_elements(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
    n = length(lie_basis)
    repr_elements = zeros(ComplexF64, n, n, n)
    for alpha in 1:n
        for beta in 1:n
            for gamma in 1:n
                repr_elements[alpha,beta,gamma] = tr(im*lie_basis[beta] * br(lie_basis[gamma], im*lie_basis[alpha]))
            end
        end
    end
    return repr_elements
end

function construct_repr_element_matrix(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}}, element::SparseMatrixCSC{ComplexF64, Int})
    n = length(lie_basis)
    repr_elements = zeros(ComplexF64, n, n)
    for alpha in 1:n
        for beta in 1:n
            repr_elements[alpha,beta] = tr(im*lie_basis[beta] * br(element, im*lie_basis[alpha]))
        end
    end
    return repr_elements
end

function transform_observable_adjoint(observable::SparseMatrixCSC{ComplexF64, Int}, lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
    coeffs = [tr(observable * im*element) for element in lie_basis]
    return coeffs
end


function observable_expectation(
    observable::SparseMatrixCSC{ComplexF64, Int}, 
    lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}},
    input::SparseMatrixCSC{Float64, Int},
    theta::Vector{Float64})

    adjoint_repr_gen1 = construct_repr_element_matrix(lie_basis, -im*lie_basis[1])
    adjoint_repr_gen2 = construct_repr_element_matrix(lie_basis, -im*lie_basis[2])
    circuit = exp(-im*(theta[1]*adjoint_repr_gen1 + theta[2]*adjoint_repr_gen2))
    adjoint_vec_in = [tr(-im*element * input) for element in lie_basis]
    adjoint_vec_out = circuit * adjoint_vec_in
    adjoint_observable_vec = [tr(observable * im*element) for element in lie_basis]
    result = dot(vec(adjoint_observable_vec), vec(adjoint_vec_out))
    return result
end