using ForwardDiff
# unitarity and hermiticity checks
check_tol = 1e-6
function check_anti_hermiticity(H)
    # H = -adjoint(H)
    @assert isapprox(H, -adjoint(H)) "Adjoint drift is not anti-hermitian"
end

function check_unitarity(U, tmp; timestep = nothing, note = nothing)
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
        c = dot(element, remainder) / dot(element, element)
        remainder .-= c * element
        println("c:$c")
        throw( "$identifier not in p-subspace. $c > $check_tol Norm of the remainder: $(norm(remainder))  at timestep $timestep")
    end 
end

function check_belongs_to_p_subspace(mat::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
    algebra::Algebra; timestep=nothing, identifier=nothing)
    return nothing
end 
