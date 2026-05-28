# Define bracket
br(A, B) = A * B - B * A
commutes(x, y; tol = 1e-6) = maximum(abs.(br(x, y))) < tol

T = ComplexF64

# this is actually orthogonal!
function try_add_orthonormal!(basis::Vector{SparseMatrixCSC{T,Int}}, 
                        candidate:: SparseMatrixCSC{T,Int};
                        tol = 1e-6)
    
    for element in basis
        # println(dot(element, element))
        proj_coeff = dot(element, candidate) # more efficient than trace
        candidate .-= proj_coeff .* element ./ dot(element, element)
    end

    nrm = norm(candidate)
    if nrm < tol * sqrt(length(candidate))
        return false
    end

    candidate ./= nrm
    push!(basis, candidate)

    return true
end

function construct_lie_basis_general(generators::Vector{SparseMatrixCSC{T, Int}}; depth = 10)
    basis_elements = SparseMatrixCSC{T,Int}[]
    # gens = [im * g for g in generators]
    gens = copy(generators)
    for g in gens
        try_add_orthonormal!(basis_elements, g)
        #push!(basis_elements, g)#/norm(g))
    end
    last_level = copy(generators)
    if depth > 1
        for d in 1:depth 
            next_level = SparseMatrixCSC{T,Int}[]
            for g in gens
                for last_el in last_level
                    bracket = br(g, last_el)
                    if try_add_orthonormal!(basis_elements, bracket)
                        push!(next_level, bracket)
                        # println(d)
                    end
                end
            end
            last_level = next_level
        end
    end
    return basis_elements
end

function project_algebra(mat, lie_basis; tol = 1e-8)
    remainder = copy(mat)
    coeffs = zeros(real(eltype(mat)),length(lie_basis))
    for (i, el) in enumerate(lie_basis)
        coeffs[i] = real(dot(el, remainder) / dot(el, el))
        remainder .-= coeffs[i] .* el
    end 
    @assert norm(remainder) < tol "element outside algebra, norm(remainder) = $(norm(remainder))"
    return coeffs
end

function project_to_algebra!(coeffs, mat, algebra, stor; tol = 1e-8, identifier=nothing)
    # orthonormal basis assumed 
    fill!(coeffs, zero(eltype(coeffs)))
    for (i, el) in enumerate(algebra.lie_basis)
        coeffs[i] = real(tr(el' * mat))
    end 
    stor.proj_alg_tmp .= mat 
    for (i, el) in enumerate(algebra.lie_basis)
        stor.proj_alg_tmp .-= coeffs[i] .* el 
    end 

    @assert norm(stor.proj_alg_tmp) < tol "element outside algebra, norm(remainder) = $(norm(stor.proj_alg_tmp)), coeffs: $coeffs. Found for $identifier"
    return nothing
end

function Lie_to_Hilbert!(res::Matrix{T}, res_arr::Vector{Float64}, algebra)
    res .= 0
    for (i,μ) in enumerate(res_arr)
        res .+= μ .* algebra.lie_basis[i]
    end 
    return nothing 
end 

function build_structure_tensor(lie_basis::Vector{SparseMatrixCSC{T, Int}}; tol=1e-10)
    # Assumes an orthonormal basis
    n = length(lie_basis)
    f = zeros(T, n, n, n)
    for a in 1:n, b in a+1:n
        comm = br(lie_basis[a], lie_basis[b])
        for c in 1:n
            val = tr(lie_basis[c]' * comm)
            f[c,a,b] = real(val)
            f[c,b,a] = -f[c,a,b]
            @assert abs(imag(val)) < tol "structure constant [$c, $a, $b] has large imaginary part: $(imag(val))"
        end 
    end 
    return f
end 

function lie_bracket_coeffs!(res::AbstractVector{T}, f::Array{Tf, 3}, x::Vector{T1}, y::AbstractVector{T2}) where {T, Tf, T1, T2}
    n = length(x)
    fill!(res, zero(T))
    for a in 1:n, b in a+1:n 
        xayb = x[a] * y[b] - x[b] * y[a]  # since f is antisymmetric
        iszero(xayb) && continue 
        for c in 1:n 
            res[c] += f[c,a,b] * xayb
        end 
    end 
    return nothing 
end 