
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
    gen1, gen2 = generators
    bracket = br(im*gen1, im*gen2)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[im*gen1, im*gen2]
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
    return basis_elements
end
