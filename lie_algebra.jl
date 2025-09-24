br(A, B) = A * B - B * A

function is_independent(matrix_list::Vector{<:SparseMatrixCSC{ComplexF64, Int}}, candidate_matrix::SparseMatrixCSC{ComplexF64, Int}; tol_rel=1e-12,tol_abs=1e-14)
    if isempty(matrix_list)
        return true
    end
    M = hcat([vec(C) for C in matrix_list]...)
    d = Array(vec(candidate_matrix))
    x = M \ d
    residual = norm(M * x - d)  / norm(d)

    return residual ≥ tol_rel

end

function orthonormalise_basis(basis::Vector{SparseMatrixCSC{ComplexF64, Int}}; tol=1e-14)
    orthonormal_basis = SparseMatrixCSC{ComplexF64, Int}[]
    for i in eachindex(basis)
        element = copy(basis[i])
        for orthonormal_element in orthonormal_basis
            proj_coeff = tr(orthonormal_element' * element)
            element .-= proj_coeff * orthonormal_element
        end
        nrm = sqrt(real(tr(element' * element)))
        if nrm < tol
            continue
        end
        push!(orthonormal_basis, element / nrm)
    end
    return orthonormal_basis
end

function construct_lie_basis(generators::Vector{SparseMatrixCSC{ComplexF64, Int}}, depth::Int)
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
    basis_elements = orthonormalise_basis(basis_elements)

    return basis_elements
end

function construct_adjoint_representations(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}},
                                           generators::Vector{SparseMatrixCSC{ComplexF64, Int}})
    n = length(lie_basis)
    adjoint_map = [zeros(ComplexF64, n, n) for _ in 1:length(generators)]

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

function gsim_expectation_value(
        observable::SparseMatrixCSC{ComplexF64, Int},
        input_matrix::SparseMatrixCSC{Float64, Int},
        theta::Vector{Float64},
        lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}},
        adjoint_map::Vector{Matrix{ComplexF64}}
    )

    v_in = Vector{eltype(observable)}(undef, length(lie_basis))
    temp_element = similar(observable) 
    for (i, element) in enumerate(lie_basis)
        mul!(temp_element, element', input_matrix)
        v_in[i] = im*tr(temp_element)
    end
    v_obs = Vector{eltype(observable)}(undef, length(lie_basis))
    temp_element = similar(observable)
    for (i, element) in enumerate(lie_basis)
        mul!(temp_element, element', observable)
        v_obs[i] = im*tr(temp_element)
    end

    A, B = adjoint_map[1], adjoint_map[2]
    tmp = similar(A)                    
    mul!(tmp, A, theta[1])              
    BLAS.axpy!(theta[2], B, tmp)      
    lmul!(-im, tmp)             
    circuit = exp(tmp)
    v_out = similar(v_in)
    mul!(v_out, circuit, v_in)

    return dot(v_out, v_obs)
end