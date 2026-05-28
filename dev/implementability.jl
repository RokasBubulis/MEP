include("generators.jl")

"Implementability tests are not correct at this moment"

# Check whether unitary target is implementable
function check_target(target::SparseMatrixCSC{float_type, Int}, system_basis::Vector{SparseMatrixCSC{float_type, Int}})

    reconstructed_target = spzeros(float_type, size(target)...)
    for i in eachindex(system_basis)
        # coeff = tr(adjoint(system_basis[i]) * target)
        coeff = dot(system_basis[i], target)
        reconstructed_target .+= coeff .* system_basis[i]
    end

    residual = reconstructed_target - target
    return norm(residual)
end

function check_if_belongs(target::SparseMatrixCSC{float_type, Int}, system_basis::Vector{SparseMatrixCSC{float_type, Int}}; tol = 1e-6)

    reconstructed_target = spzeros(float_type, size(target)...)
    for i in eachindex(system_basis)
        coeff = dot(system_basis[i], target)
        reconstructed_target .+= coeff .* system_basis[i]
    end

    residual = reconstructed_target - target
    return norm(residual) < tol
end

# wrong targets
# function construct_CZ_target(n_qubits::Int, n_levels::Int)
#     dim = n_levels^n_qubits
#     U = SparseMatrixCSC{ComplexF64}(I, dim, dim)  # true identity
#     U[dim, dim] = -1 + 0im
#     return U
# end

# function construct_target_3levels(n_qubits::Int)
#     I3 = sparse(float_type[1 0 0; 0 1 0; 0 0 1])
#     target = copy(I3)
#     for n in 2:n_qubits
#         target = kron(target, I3)
#     end
#     target[3^n_qubits, 3^n_qubits] = -1.0
#     return target
# end

# function construct_target_2levels(n_qubits::Int)
#     I2 = sparse(float_type[1 0; 0 1])
#     target = copy(I2)
#     for n in 2:n_qubits
#         target = kron(target, I2)
#     end
#     target[2^n_qubits, 2^n_qubits] = -1.0
#     return target
# end

function check_if_implementable(basis::Vector{SparseMatrixCSC{float_type, Int}}, 
    target::SparseMatrixCSC{float_type, Int}; max_product_depth = 10, print_output = false, tol=1e-6)

    @assert max_product_depth >= 2 "Increase max product depth"
    res_norm, last_layer = 0,0
    basis_elements = SparseMatrixCSC{float_type,Int}[]
    n = size(basis[1], 1)
    try_add_orthonormal!(basis_elements, spdiagm(0 => ones(float_type, n)))
    for b in basis
        try_add_orthonormal!(basis_elements, b)
    end

    last_level = copy(basis)
    new_el = similar(basis[1])
    for d in 2:max_product_depth
        current_level = SparseMatrixCSC{float_type,Int}[]
        for el in last_level
            for b_el in basis
                new_el = el * b_el
                if try_add_orthonormal!(basis_elements, new_el)
                    push!(current_level, new_el)
                end
            end
        end
        last_level = current_level
        res_norm = check_target(target, basis_elements)
        last_layer = d
        if res_norm < tol
            break
        end
    end

    if print_output
        return ("Subalgebra dim: $(length(basis)), Subgroup dim: $(length(basis_elements)), Residual norm: $res_norm, required product length: $last_layer")
    else
        return res_norm < tol
    end
end


function construct_CZ_target(n_qubits::Int, n_levels::Int)
    dim = n_levels^n_qubits
    U = SparseMatrixCSC{ComplexF64}(I, dim, dim)  # true identity
    U[5, 5] = -1 + 0im
    return U
end

function check_group_implementability(target::SparseMatrixCSC{float_type, Int}, lie_basis::Vector{SparseMatrixCSC{float_type, Int}})
    choices= 0
    n = size(target, 1)
    combs = [vcat(collect(t), 0) for t in Iterators.product(ntuple(_ -> choices, n-1)...)]
    #combs = [collect(t) for t in Iterators.product(ntuple(_ -> choices, n)...)]
    # lie basis is orthonormal
    θ = angle.(diag(target))
    min_residual = Inf
    best_residual_matrix = similar(target)
    for comb in combs

        phases = θ .+ 2*pi .*comb
        ln_target = spdiagm(0=> im.*phases)
        ln_target .-= (tr(ln_target)/n) * spdiagm(0 => ones(float_type, n))   

        reconstructed_target = spzeros(float_type, size(target)...)
        for i in eachindex(lie_basis)
            coeff = dot(lie_basis[i], ln_target)
            reconstructed_target .+= coeff .* lie_basis[i]
        end

        residual_norm = norm(reconstructed_target - ln_target)
        if residual_norm < min_residual
            min_residual = residual_norm
            best_residual_matrix = reconstructed_target - ln_target
        end
        # min_residual = min(min_residual, residual_norm)

        if residual_norm < 1e-6
            println("Implementability check passed")
            return true
        end
    end
    display(best_residual_matrix)
    println("Implementability check failed with minimum residual: $min_residual")
    return false
end