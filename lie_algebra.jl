# Define bracket
br(A, B) = A * B - B * A
commutes(x, y; tol = 1e-6) = maximum(abs.(br(x, y))) < tol

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
function construct_lie_basis_2gens(generators::Vector{SparseMatrixCSC{float_type, Int}}, depth::Int)
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

function construct_subgroup_basis(lie_basis::Vector{SparseMatrixCSC{float_type, Int}}, max_product_depth::Int)
    basis_elements = SparseMatrixCSC{float_type,Int}[]
    n = size(lie_basis[1], 1)
    try_add_orthonormal!(basis_elements, spdiagm(0 => ones(float_type, n)))

    for b in lie_basis
        try_add_orthonormal!(basis_elements, b)
    end
    last_level = copy(lie_basis)
    if max_product_depth > 1
        for d in 2:max_product_depth
            old_len = length(basis_elements)
            current_level = SparseMatrixCSC{float_type,Int}[]
            for last_element in last_level
                for b_element in lie_basis
                    new_el = last_element * b_element
                    if try_add_orthonormal!(basis_elements, new_el)
                        push!(current_level, new_el)  # copy only when accepted
                    end
                end
            end
            last_level = current_level
            if length(basis_elements) == old_len
                break
            end
        end
    end
    return basis_elements
end

function bron_kerbosch_recursive(p_c, edges, n)
    drift_idx = 1 #drift index in p_c is by assertion 1
    R = [drift_idx]  # clique must start with drift
    P = [i for i in 1:n if edges[drift_idx, i] && i != drift_idx] # drift neighbours
    X = Int[] # processed nodes
    cliques = Vector{Vector{Int}}()
    
    function bronk!(R, P, X)
        if isempty(P) && isempty(X)
            push!(cliques, copy(R))
            return 
        end

        for node in copy(P)
            neighbours = [i for i in 1:n if edges[node, i]]
            bronk!(vcat(R, node), intersect(P, neighbours), intersect(X, neighbours))
            P = setdiff(P, [node])
            push!(X, node)
        end
    end
    bronk!(R, P, X)
    max_clique = cliques[argmax(length.(cliques))]
    return [p_c[i] for i in max_clique]
end

function construct_max_abelian_subalgebra(p_c)
    n = length(p_c)
    edges = falses(n,n)

    for i in 1:n-1
        for j in i+1:n
            if commutes(p_c[i], p_c[j])
                edges[i, j] = true
                edges[j, i] = true
            end
        end
    end
    a_c = bron_kerbosch_recursive(p_c, edges, n)
    return a_c
end


function construct_algebras(drift, controls; commutation_depth = 10, tol = 1e-12)

    l_c = construct_lie_basis_general(controls, commutation_depth)
    # remove the component of the drift which is part of the control lie algebra
    corrected_drift = copy(drift)
    for b in l_c
        corrected_drift .-= dot(b, corrected_drift) * b
    end
    if all(x->abs(dot(corrected_drift, x)) < tol, l_c) 
        p_c = SparseMatrixCSC{float_type,Int}[]
    else
        error("No drift component in the orthogonal complement")
    end

    gens = SparseMatrixCSC{float_type, Int}[corrected_drift, controls...]
    g_c = construct_lie_basis_general(gens, commutation_depth)
    for g in g_c
        if all(x -> abs(dot(g, x)) < tol, l_c)
            try_add_orthonormal!(p_c, g)
        end
    end
    @assert p_c[1] == g_c[1] "Corrected drift must be the first element of the orthogonal complement"
    a_c = construct_max_abelian_subalgebra(p_c)

    # a_c = SparseMatrixCSC{float_type,Int}[]
    # candidates = copy(p_c)
    # #Construct a_c
    # while !isempty(candidates)
    #     new_candidates = SparseMatrixCSC{float_type,Int}[]
    #     old_length = length(a_c)
    #     for p_el in candidates
    #         if all(x -> commutes(x, p_el), a_c) #&& !any(x -> x == p_el, a_c)
    #             push!(a_c, p_el)
    #         else
    #             push!(new_candidates, p_el)
    #         end
    #     end
    #     if length(a_c) == old_length
    #         break
    #     end
    #     candidates = new_candidates
    # end

    return g_c, l_c, p_c, a_c
end


function construct_weyl_group_orbit(control_subgroup, max_abelian_subalgebra; tol=1e-3)

    # Construct weyl group first
    weyl_group = SparseMatrixCSC{float_type,Int}[]
    for K in control_subgroup
        is_suitable = true
        for a in max_abelian_subalgebra
            KaK = K * a * adjoint(K)  # K is unitary hence adjoint is inverse
            if !check_if_belongs(KaK, max_abelian_subalgebra)
                is_suitable = false
                break
            end

            if norm(KaK - a) < tol
                is_suitable = false
                break
            end
        end
        if is_suitable
            push!(weyl_group, K)
        end
    end

    # Now construct the orbit 
    weyl_group_orbit = SparseMatrixCSC{float_type,Int}[]
    for a in max_abelian_subalgebra
        for w in weyl_group
            WaW = w * a * adjoint(w)
            if !(WaW in weyl_group_orbit)
                push!(weyl_group_orbit, WaW)
            end
        end
    end

    return weyl_group_orbit
end