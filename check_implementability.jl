using LinearAlgebra

function check_target(target::SparseMatrixCSC{ComplexF64, Int}, system_basis::Vector{SparseMatrixCSC{ComplexF64, Int}})

    reconstructed_target = spzeros(ComplexF64, size(target)...)
    for i in eachindex(system_basis)
        coeff = tr(adjoint(system_basis[i]) * target)
        reconstructed_target += coeff * system_basis[i]
    end

    residual = reconstructed_target - target
    return norm(residual), residual
end

function construct_target(n_qubits::Int)
    I3 = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])
    target = copy(I3)
    for n in 2:n_qubits
        target = kron(target, I3)
    end
    target[3^n_qubits, 3^n_qubits] = exp(im*sqrt(3)/5*pi)
    return target
end

function construct_subgroup_basis(lie_basis::Vector{SparseMatrixCSC{ComplexF64, Int}}, product_depth::Int)
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[]
    n = size(lie_basis[1], 1)
    try_add_orthonormal!(basis_elements, spdiagm(0 => ones(ComplexF64, n)))

    for b in lie_basis
        try_add_orthonormal!(basis_elements, b)
    end
    last_level = copy(lie_basis)
    if product_depth > 1
        tmp = similar(lie_basis[1])  # allocate once
        for d in 2:product_depth

            current_level = SparseMatrixCSC{ComplexF64,Int}[]
            for last_element in last_level
                for b_element in lie_basis
                    mul!(tmp, last_element, b_element)
                    if try_add_orthonormal!(basis_elements, tmp)
                        push!(current_level, copy(tmp))  # copy only when accepted
                    end
                end
            end
            last_level = current_level
        end
    end

    return basis_elements
end

function check_if_implementable(lie_basis::Vector{SparseMatrixCSC{ComplexF64, Int}}, 
    unitary_target::SparseMatrixCSC{ComplexF64, Int}, max_product_depth::Int; tol=1e-6)

    @assert max_product_depth >= 2 "Increase max product depth"
    res_norm, last_layer = 0,0
    res = similar(lie_basis[1])
    basis_elements = SparseMatrixCSC{ComplexF64,Int}[]
    n = size(lie_basis[1], 1)
    try_add_orthonormal!(basis_elements, spdiagm(0 => ones(ComplexF64, n)))
    for b in lie_basis
        try_add_orthonormal!(basis_elements, b)
    end

    last_level = copy(lie_basis)
    new_el = similar(lie_basis[1])
    for d in 2:max_product_depth
        current_level = SparseMatrixCSC{ComplexF64,Int}[]
        for el in last_level
            for b_el in lie_basis
                new_el = el * b_el
                if try_add_orthonormal!(basis_elements, new_el)
                    push!(current_level, new_el)
                end
            end
        end
        last_level = current_level
        res_norm, res = check_target(unitary_target, basis_elements)
        last_layer = d
        if res_norm < tol
            break
        end
    end
    return ("Subalgebra dim: $(length(lie_basis)), Subgroup dim: $(length(basis_elements)), Residual norm: $res_norm, required product length: $last_layer")
end