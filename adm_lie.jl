
function adjoint_action_by_campbell_structure_tensor_arr!(res_arr::Vector{T}, x_arr::Vector{T}, y_arr::Vector{T}, algebra::Algebra, stor::Storage; depth = 20) where T

    last_term_lie_coeffs = stor.campbell_array3
    new_term_lie_coeffs = stor.campbell_array4

    last_term_lie_coeffs .= y_arr
    res_arr .= y_arr
    coeff = one(eltype(res_arr))

    for n in 1:depth
        lie_bracket_coeffs!(new_term_lie_coeffs, algebra.structure_tensor, x_arr, last_term_lie_coeffs)
        coeff /= n 
        res_arr .+= coeff .* new_term_lie_coeffs
        last_term_lie_coeffs .= new_term_lie_coeffs
    end
    return nothing
end 

function adjoint_drift_arr!(res_arr::Vector{T}, α::T, im_control_arr::Vector{T}, neg_im_drift_arr::Vector{T}, algebra::Algebra, stor::Storage) where T
    im_control_arr .*= -α
    adjoint_action_by_campbell_structure_tensor_arr!(res_arr, im_control_arr, neg_im_drift_arr, algebra, stor)
    return nothing 
end 

function Lie_to_Hilbert!(res::Matrix{T}, res_arr::Vector{T}, algebra::Algebra)
    res .= 0
    for (i,μ) in enumerate(res_arr)
        res .+= μ .* algebra.lie_basis[i]
    end 
    return nothing 
end 

function project_to_algebra!(coeffs::Vector{Float64}, mat::Matrix{T}, algebra, stor; tol = 1e-8, identifier=nothing) where T
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

function project_to_lie_basis(mat, lie_basis; tol = 1e-8)
    # orthonormal basis assumed 
    coeffs = zeros(Float64, length(lie_basis))
    for (i, el) in enumerate(lie_basis)
        coeffs[i] = real(tr(el' * mat))
    end 
    mat_copy = copy(mat) 
    for (i, el) in enumerate(lie_basis)
        mat_copy .-= coeffs[i] .* el 
    end 

    @assert norm(mat_copy) < tol "element outside algebra, norm(remainder) = $(norm(mat_copy)), coeffs: $coeffs"
    return coeffs
end

function adjoint_drift_obj_arr(α::Float64, costate_arr::Vector{Float64}, im_control_arr::Vector{Float64}, neg_im_drift_arr::Vector{Float64}, algebra::Algebra, stor::Storage)
    adjoint_drift_arr!(stor.tmp_adj_drift_arr, α, im_control_arr, neg_im_drift_arr, algebra, stor)
    return -dot(stor.tmp_adj_drift1_arr, costate_arr)
end

function optimal_adjoint_drift_optimiser_arr!(res_arr::Vector{T}, costate_arr::Vector{T}, algebra::Algebra, stor::Storage) where T
    x0 = [0.0]
    obj(x) = -adjoint_drift_obj_arr(x[], costate_arr, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra, stor)
    res = Optim.optimize(obj, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[]
    adjoint_drift_arr!(res_arr, α_optimal, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra, stor)

    if abs(α_optimal) > 8.0
        @warn("|α_opt| = $α_optimal > 8.0 while Campbell identity is used, consider changing series depth")
    end 

    return nothing 
end 

optimal_adjoint_drift!(tmp::Vector{Float64}, costate::Vector{Float64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_optimiser_arr!(tmp, costate, algebra, stor)