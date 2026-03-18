
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