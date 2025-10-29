# Default matrix exponentiation approach
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

# Lie-algebraic approach
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

# Tol might need to be increased if going above 8 qubits
function try_add_orthonormal!(basis::Vector{SparseMatrixCSC{ComplexF64,Int}}, 
                        candidate:: SparseMatrixCSC{ComplexF64,Int};
                        tol=1e-5)

    for element in basis
        proj_coeff = tr(adjoint(element) * candidate)
        candidate = candidate - proj_coeff*element
    end
    nrm = norm(candidate)
    if nrm < tol
        return false
    end
    push!(basis, candidate/nrm)
    return true
end

# Local GS orthonormalisation
function construct_lie_basis_fast(generators::Vector{SparseMatrixCSC{ComplexF64, Int}}, depth::Int)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[]
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
        next_layer = SparseMatrixCSC{ComplexF64, Int}[]
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
        adjoint_map::Vector{Matrix{ComplexF64}})

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