using ForwardDiff
# unitarity and hermiticity checks
check_tol = 1e-6
function check_anti_hermiticity(H)
    # H = -adjoint(H)
    @assert isapprox(H, -adjoint(H)) "Adjoint drift is not anti-hermitian"
end

function is_dual_matrix(U)
    T = eltype(U)
    return (T <: Complex && real(T) <: ForwardDiff.Dual)
end

function check_unitarity(U, tmp; timestep = nothing, note = nothing)
    is_dual_matrix(U) && return
    if any(isnan, U)
        error("$(note !== nothing ? note : "") NaN in propagator at timestep $timestep")
    end
    # U*adjoint(U) = I
    mul!(tmp, U, adjoint(U))
    nrm = norm(tmp) - sqrt(size(U,1))
    @assert nrm < check_tol "$(note !== nothing ? note : "") Propagator is not unitary at timestep $timestep: norm(U*adjoint(U) - I) = $nrm"
end

function check_belongs_to_p_subspace(mat::Union{Matrix{T}, SparseMatrixCSC{T, Int}}, 
    algebra::Algebra; timestep = nothing, identifier = nothing) where {T<:Number}
    remainder = copy(mat)
    for element in algebra.p_basis
        coeff = dot(element, remainder) / dot(element, element)
        remainder .-= coeff .* element
    end 
    if !isapprox(norm(remainder), 0.0, atol=check_tol)
        element = algebra.lie_basis[1]
        c = real(dot(element, remainder) / dot(element, element))
        # remainder .-= c * element
        throw("$identifier not in p-subspace. Norm of the remainder: $(norm(remainder)) at timestep $timestep. Overlap with control: $c")
    end 
end

function check_belongs_to_p_subspace(mat::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
    algebra::Algebra; timestep=nothing, identifier=nothing)
    return nothing
end 
