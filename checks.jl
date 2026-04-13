
# unitarity and hermiticity checks
unitarity_tol = 1e-6
function check_anti_hermiticity(H)
    # H = -adjoint(H)
    @assert isapprox(H, -adjoint(H)) "Adjoint drift is not anti-hermitian"
end

function check_unitarity(U, i; note = nothing)
    # U*adjoint(U) = I
    nrm = norm(U*adjoint(U)) - sqrt(size(U,1))
    @assert nrm < unitarity_tol "$(note !== nothing ? note : "") Propagator is not unitary at timestep $i: norm(U*adjoint(U) - I) = $nrm"
end

function check_costate(mat1, params, timestep)
    n = length(params.algebra.p_basis)
    dim = size(params.physics.target,1)
    mat = copy(mat1)
    for element in params.algebra.p_basis
        coeff = dot(element, mat) / dot(element, element)
        mat .-= coeff .* element
    end 
    @assert isapprox(norm(mat), 0.0, atol=1e-8) "norm of remainder: $(norm(mat)) at timestep $timestep"
end