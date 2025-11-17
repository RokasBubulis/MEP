include("generators.jl")
include("lie_algebra.jl")
include("check_implementability.jl")

n_qubits = 2
commutation_depth = 10
product_depth = 5
gens = construct_coupled_spin_generators(n_qubits)
drift, controls = gens[1], gens[2:end]

g_c = construct_lie_basis_general(gens, commutation_depth)
l_c = construct_lie_basis_general(controls, commutation_depth)

p_c = SparseMatrixCSC{float_type,Int}[]
for el in g_c
    if !(el in l_c)
        push!(p_c, el)
    end
end


function construct_max_abelian_subalgebra(p_c, drift)
    a_c = SparseMatrixCSC{float_type,Int}[]
    push!(a_c, im*drift)

    candidates = copy(p_c)

    while !isempty(candidates)
        new_candidates = SparseMatrixCSC{float_type,Int}[]
        old_length = length(a_c)
        for p_el in candidates
            if all(x -> nnz(br(x, p_el)) == 0, a_c) && !any(x -> x == p_el, a_c)
                push!(a_c, p_el)
            else
                push!(new_candidates, p_el)
            end
        end
        if length(a_c) == old_length
            break
        end
        candidates = new_candidates
    end

    return a_c
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

a_c = construct_max_abelian_subalgebra(p_c, drift)

println("len(g_c) = $(length(g_c))")
println("len(l_c) = $(length(l_c))")
println("len(p_c) = $(length(p_c))")
println("len(a_c) = $(length(a_c))")

control_subgroup_basis = construct_subgroup_basis(l_c, product_depth)
println("len(l_c subgroup) = $(length(control_subgroup_basis))")
weyl_group_orbit = construct_weyl_group_orbit(control_subgroup_basis, a_c)
println("len(W orbit) = $(length(weyl_group_orbit))")
println("Orbit elements:")
for item in weyl_group_orbit
    display(item)
end
println("---")
display(im*operator(Xop([1,2]),2))
display(im*operator(Yop([1,2]),2))
display(im*operator(Zop([1,2]),2))