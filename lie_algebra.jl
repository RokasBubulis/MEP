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

function orthonormalize_sparse_basis(B::Vector{SparseMatrixCSC{ComplexF64, Int}}; tol=1e-14)
    Q = SparseMatrixCSC{ComplexF64, Int}[]
    for i in eachindex(B)
        v = copy(B[i])
        for q in Q
            α = tr(q' * v)
            v .-= α * q
        end
        nrm = sqrt(real(tr(v' * v)))
        if nrm < tol
            continue
        end
        push!(Q, v / nrm)
    end
    return Q
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
    # basis_elements = [X / norm(X) for X in basis_elements]
    basis_elements = orthonormalize_sparse_basis(basis_elements)

    return basis_elements
end

# function construct_repr_elements(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}})
#     n = length(lie_basis)
#     repr_elements = zeros(ComplexF64, n, n, n)
#     for alpha in 1:n
#         for beta in 1:n
#             for gamma in 1:n
#                 repr_elements[alpha,beta,gamma] = tr(im*lie_basis[beta] * br(lie_basis[gamma], im*lie_basis[alpha]))
#             end
#         end
#     end
#     return repr_elements
# end

# function construct_structure_tensor(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}}, 
#                                     generators::Vector{SparseMatrixCSC{ComplexF64, Int}})
#     n = length(lie_basis)
#     adjoint_map = zeros(ComplexF64, n, n, 2)
#     for alpha in 1:n
#         for beta in 1:n
#             for gamma in 1:2
#                 adjoint_map[alpha,beta, gamma] = tr((im*lie_basis[beta] * br(im*generators[gamma], im*lie_basis[alpha])))
#             end
#         end
#     end
#     return adjoint_map
# end

# function construct_adjoint_representations(lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}}, 
#                                     generators::Vector{SparseMatrixCSC{ComplexF64, Int}})
#     n = length(lie_basis)
#     adjoint_map_1 = zeros(ComplexF64, n, n)
#     adjoint_map_2 = zeros(ComplexF64, n, n)
#     for alpha in 1:n
#         for beta in 1:n
#             adjoint_map_1[alpha, beta] = tr(im*lie_basis[beta] * br(im*generators[1], im*lie_basis[alpha]))
#             adjoint_map_2[alpha, beta] = tr(im*lie_basis[beta] * br(im*generators[2], im*lie_basis[alpha]))
#         end
#     end
#     return [adjoint_map_1, adjoint_map_2]

# end

# function gsim_expectation_value(
#     observable::SparseMatrixCSC{ComplexF64, Int}, 
#     input_matrix::SparseMatrixCSC{Float64, Int}, 
#     theta::Vector{Float64},
#     lie_basis::Vector{SparseMatrixCSC{ComplexF64,Int}},
#     adjoint_map::Vector{Matrix{ComplexF64}})

#     hermitian_lie_basis  = im*lie_basis
#     println("structure tensor 1")
#     pretty_table(adjoint_map[1])
#     println("structure tensor 2")
#     pretty_table(adjoint_map[2])

#     circuit = exp(-im*(theta[1]*adjoint_map[1])) * exp(-im*(theta[2]*adjoint_map[2]))
#     println("circuit")
#     pretty_table(circuit)

#     # adjoint_vec_in = Vector{eltype(observable)}(undef, length(lie_basis))
#     # temp_element = similar(observable) 
#     # for (i, element) in enumerate(lie_basis)
#     #     mul!(temp_element, element', input_matrix)
#     #     adjoint_vec_in[i] = im*tr(temp_element)
#     # end
#     adjoint_vec_in = [tr(element'*input_matrix) for element in hermitian_lie_basis]

#     # println("ad vec in")
#     # pretty_table(adjoint_vec_in)
#     # adjoint_vec_out = similar(adjoint_vec_in)
#     # mul!(adjoint_vec_out, circuit, adjoint_vec_in)
#     adjoint_vec_out = circuit * adjoint_vec_in

#     # adjoint_observable_vec = Vector{eltype(observable)}(undef, length(lie_basis))
#     # temp_el = similar(observable) 
#     # for (i, element) in enumerate(lie_basis)
#     #     mul!(temp_el, observable, element)
#     #     adjoint_observable_vec[i] = im*tr(temp_el)
#     # end
#     adjoint_observable_vec = [tr(observable'*element) for element in hermitian_lie_basis]

#     result = dot(adjoint_observable_vec, adjoint_vec_out)
#     println("ad vec out")
#     pretty_table(adjoint_vec_out)
#     println("ad obs vec")
#     pretty_table(adjoint_observable_vec)

#     return result
# end

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

    n = length(lie_basis)
    v_in = [tr((im*lie_basis[j])' * input_matrix) for j in 1:n]
    v_obs = [tr((im*lie_basis[j])' * observable) for j in 1:n]

    A, B = adjoint_map[1], adjoint_map[2]
    circuit = exp(-im*(theta[1]*A)) * exp(-im*(theta[2]*B))
    v_out = circuit * v_in
    result = sum(v_obs .* v_out)

    # println("adj repr of A")
    # pretty_table(A)
    # println("adj repr of B")
    # pretty_table(B)
    # println("Circuit")
    # pretty_table(circuit)
    # println("Adj v in")
    # pretty_table(v_in)
    # println("Adj v out")
    # pretty_table(v_out)
    # println("Obs v")
    # pretty_table(v_obs)

    return result
end