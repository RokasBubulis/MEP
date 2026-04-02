
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

function check_belongs_to_p_basis(mat1, params, timestep)
    n = length(params.derived_args.p_basis)
    dim = size(params.system_params.target,1)
    mat = copy(mat1)
    for i in 1:n
        coeff = dot(params.derived_args.p_basis[i], mat) / dot(params.derived_args.p_basis[i],params.derived_args.p_basis[i])
        mat .-= coeff .* params.derived_args.p_basis[i]
    end
    @assert isapprox(norm(mat), 0.0, atol=1e-3) "norm of remainder: $(norm(mat)) at timestep $timestep"
end