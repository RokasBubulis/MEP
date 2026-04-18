
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
    #nrm = norm(U*adjoint(U)) - sqrt(size(U,1))
    @assert nrm < check_tol "$(note !== nothing ? note : "") Propagator is not unitary at timestep $timestep: norm(U*adjoint(U) - I) = $nrm"
end

function check_costate(mat1, params, timestep)
    n = length(params.algebra.p_basis)
    dim = size(params.physics.target,1)
    mat = copy(mat1)
    for element in params.algebra.p_basis
        coeff = dot(element, mat) / dot(element, element)
        mat .-= coeff .* element
    end 
    @assert isapprox(norm(mat), 0.0, atol=check_tol) "norm of remainder: $(norm(mat)) at timestep $timestep"
end