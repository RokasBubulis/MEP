
function adjoint_action_by_campbell_structure_tensor_arr!(res_arr, x_arr, y_arr, algebra::Algebra, stor::Storage; depth = 20)

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

function adjoint_drift_arr!(res_arr::Vector{T}, α::Float64, im_control_arr::Vector{T}, neg_im_drift_arr::Vector{T}, algebra::Algebra, stor::Storage) where T
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

function adjoint_drift_obj_arr(α::TR, costate_arr::Vector{T}, im_control_arr::Vector{T}, neg_im_drift_arr::Vector{T}, algebra::Algebra, stor::Storage) where {TR, T}
    adjoint_drift_arr!(stor.tmp_adj_drift_arr, α, im_control_arr, neg_im_drift_arr, algebra, stor)
    return real(-sum(stor.tmp_adj_drift1_arr[i] * costate_arr[i] for i in eachindex(costate_arr)))
end

# function adjoint_drift_obj_1st_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

#     adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
#     bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for first: ")
#     if real(eltype(costate)) <: ForwardDiff.Dual 
#         res = stor.tmp_adjoint_drift_1st_der_obj_dual
#     else
#         res = stor.tmp_adjoint_drift_1st_der_obj
#     end 
#     mul!(res, stor.tmp_adjoint_drift_1st_der, costate)
#     return real(tr(res))
# end 

# function adjoint_drift_obj_2nd_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

#     adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
#     bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for second: ")
#     bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_2nd_der, stor.tmp_adjoint_drift_1st_der, system.im_control, algebra, stor; identifier="Second der: ")
#     mul!(stor.tmp_adjoint_drift_2nd_der_obj, stor.tmp_adjoint_drift_2nd_der, costate)
#     return real(tr(stor.tmp_adjoint_drift_2nd_der_obj))
# end

function optimal_adjoint_drift_optimiser_arr!(res_arr::Vector{T}, costate_arr::Vector{T}, algebra::Algebra, system::System, stor::Storage) where T
    x0 = [0.0]

    # td = TwiceDifferentiable(
    # x -> -adjoint_drift_obj(x[], costate, algebra, solver, stor),
    # (G, x) -> (G .= -adjoint_drift_obj_1st_der(x[], costate, algebra, system, solver, stor)),
    # (H, x) -> (H .= -adjoint_drift_obj_2nd_der(x[], costate, algebra, system, solver, stor)),
    # x0
    # )

    obj(x) = -adjoint_drift_obj_arr(x[], costate_arr, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra, stor)
    res = Optim.optimize(obj, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[]

    adjoint_drift_arr!(res_arr, α_optimal, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra, stor)
    # # ensure optimal adjoint drift is anti-hermitian
    # check_anti_hermiticity(tmp)

    return nothing 
end 

optimal_adjoint_drift!(tmp::Vector{ComplexF64}, costate::Vector{ComplexF64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_optimiser_arr!(tmp, costate, algebra, system, stor)