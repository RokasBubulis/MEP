# Define bracket
br(A, B) = A * B - B * A

function try_add_orthonormal!(basis::Vector{SparseMatrixCSC{float_type,Int}}, 
                        candidate:: SparseMatrixCSC{float_type,Int};
                        tol = 1e-8)
    
    for element in basis
        proj_coeff = dot(element, candidate) # more efficient than trace
        candidate .-= proj_coeff .* element
    end

    nrm = norm(candidate)
    if nrm < tol * sqrt(length(candidate))
        return false
    end

    candidate ./= nrm
    push!(basis, candidate)

    return true
end

# Local GS orthonormalisation
function construct_lie_basis(generators::Vector{SparseMatrixCSC{float_type, Int}}, depth::Int)
    basis_elements = SparseMatrixCSC{float_type,Int}[]
    im_gens = im.*generators
    for g in im_gens
        try_add_orthonormal!(basis_elements, g)
    end
    bracket = br(im_gens[1], im_gens[2])
    if try_add_orthonormal!(basis_elements, bracket)
        new_elements = [bracket]
    else
        return basis_elements
    end
    d = 1
    while d < depth
        d += 1
        next_layer = SparseMatrixCSC{float_type, Int}[]
        for element in new_elements
            for gen in im_gens
                bracket = br(gen, element)
                if try_add_orthonormal!(basis_elements, bracket)
                    push!(next_layer, bracket)
                end
            end
        end
        new_elements = next_layer
    end
    return basis_elements
end

function construct_lie_basis_general(generators::Vector{SparseMatrixCSC{float_type, Int}}, depth::Int)
    basis_elements = SparseMatrixCSC{float_type,Int}[]
    generators .*= im
    for g in generators
        try_add_orthonormal!(basis_elements, g)
    end
    last_level = copy(generators)
    if depth > 1
        for d in 1:depth 
            next_level = SparseMatrixCSC{float_type,Int}[]
            for gen in generators
                for last_el in last_level
                    bracket = br(gen, last_el)
                    if try_add_orthonormal!(basis_elements, bracket)
                        push!(next_level, bracket)
                    end
                end
            end
            last_level = next_level
        end
    end
    return basis_elements
end

function construct_adjoint_representations(lie_basis::Vector{SparseMatrixCSC{float_type,Int}},
                                           generators::Vector{SparseMatrixCSC{float_type, Int}})
    n = length(lie_basis)
    adjoint_map = [zeros(float_type, n, n) for _ in 1:length(generators)]

    for (gidx, g) in enumerate(generators)
        for j in 1:n
            commutator = br(g, im*lie_basis[j])
            for i in 1:n
                adjoint_map[gidx][i, j] = tr((im*lie_basis[i])' * commutator) 
            end
        end
    end
    return adjoint_map
end