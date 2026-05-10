using Optim, LineSearches

include("generators.jl")
include("checks.jl")

function br!(res, A, B, tmp)
    mul!(res, A, B)
    mul!(tmp, B, A)
    res .-= tmp
    return nothing
end 

# depth of 20 accurate to machine precision
function adjoint_action_by_campbell!(res, X::SparseMatrixCSC{TX, Int}, 
    Y::SparseMatrixCSC{TY, Int}, stor::Storage; depth = 20) where {TX, TY}
    # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
    # X = -α i control, Y = -i drift
    # both control, drift are assumed orthonormal 

    # stor.tmp1 = last_term, stor.tmp2 = new_term

    res .= Y
    stor.tmp1 .= Y
    coeff = one(eltype(res))

    for n in 1:depth
        br!(stor.tmp2, X, stor.tmp1, stor.tmp3)
        coeff /= n
        axpy!(coeff, stor.tmp2, res)
        stor.tmp1 .= copy(stor.tmp2)
    end
    return nothing
end

function adjoint_action_by_campbell_structure_tensor!(res, X::SparseMatrixCSC{TX, Int}, 
    Y::SparseMatrixCSC{TY, Int}, algebra::Algebra, stor::Storage; depth = 20) where {TX, TY}

    x_lie_coeffs = stor.campbell_array1
    y_lie_coeffs = stor.campbell_array2
    last_term_lie_coeffs = stor.campbell_array3
    new_term_lie_coeffs = stor.campbell_array4
    res_lie_coeffs = stor.campbell_array5

    project_to_algebra!(x_lie_coeffs, X, algebra, stor; identifier="Control like")
    project_to_algebra!(y_lie_coeffs, Y, algebra, stor; identifier="Drift like")

    last_term_lie_coeffs .= y_lie_coeffs
    res_lie_coeffs .= y_lie_coeffs
    coeff = one(eltype(res_lie_coeffs))

    for n in 1:depth
        lie_bracket_coeffs!(new_term_lie_coeffs, algebra.structure_tensor, x_lie_coeffs, last_term_lie_coeffs)
        coeff /= n 
        res_lie_coeffs .+= coeff .* new_term_lie_coeffs
        last_term_lie_coeffs .= new_term_lie_coeffs
    end

    fill!(res, zero(eltype(res)))
    for c in eachindex(res_lie_coeffs)
        res .+= res_lie_coeffs[c] .* algebra.lie_basis[c]
    end 
    # try 
    #     project_to_algebra!(stor.campbell_array2, res, algebra, stor; identifier="TEST")
    # catch e 
    #     println("x_lie: $x_lie_coeffs")
    #     println("y_lie: $y_lie_coeffs")
    #     println("last_term_lie_coeffs: $last_term_lie_coeffs")
    #     println("new_term_lie_coeffs: $new_term_lie_coeffs")
    #     println("res_lie_coeffs: $res_lie_coeffs")

    #     rethrow(e)
    # end 
    return nothing

end 

function  adjoint_action_true(X, Y)
    return exp(Matrix(X)) * Y * exp(-Matrix(X))
end


function adjoint_drift!(res::Matrix{TCostate}, α::TAlpha, algebra::Algebra, system::System, stor::Storage) where {TAlpha, TCostate}
    adjoint_action_by_campbell_structure_tensor!(res, -α * system.im_control, -system.im_drift, algebra, stor)
    return nothing 
end 

function adjoint_drift_obj(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}
    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    mul!(stor.tmp_adjoint_drift_obj, stor.tmp_adjoint_drift, costate)
    return real(tr(stor.tmp_adjoint_drift_obj)) - solver.lambda * α^2
end 

function adjoint_drift_obj_1st_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for first: ")

    mul!(stor.tmp_adjoint_drift_1st_der_obj, stor.tmp_adjoint_drift_1st_der, costate)
    return real(tr(stor.tmp_adjoint_drift_1st_der_obj)) - solver.lambda * 2*α
end 

function adjoint_drift_obj_2nd_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for second: ")
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_2nd_der, stor.tmp_adjoint_drift_1st_der, system.im_control, algebra, stor; identifier="Second der: ")
    mul!(stor.tmp_adjoint_drift_2nd_der_obj, stor.tmp_adjoint_drift_2nd_der, costate)
    return real(tr(stor.tmp_adjoint_drift_2nd_der_obj)) - solver.lambda * 2
end 

function differentiable_mod(α, system)
    # 1/sqrt(5) are the eigenvalues of diagonal L
    α_mod = system.period_im_control / 2π * angle(exp(2π * im * α / system.period_im_control))
    return α_mod 
end 

function optimal_adjoint_drift_analytic!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where TCostate

    for _ in 1:solver.Newton_steps
        first_der = adjoint_drift_obj_1st_der(stor.alpha, costate, algebra, system, solver, stor)
        second_der = adjoint_drift_obj_2nd_der(stor.alpha, costate, algebra, system, solver, stor)
        dα = solver.Newton_damping * first_der / second_der
        if second_der < 0
            stor.alpha -= dα
        elseif second_der > 0
            stor.alpha += dα
        end 

        stor.alpha = differentiable_mod(stor.alpha, system)

        abs(dα) < solver.Newton_tol && break
    end 

    if abs(stor.alpha) > 8.0 # alpha=8 corresponds to an error of order -9
        @warn("Unusually large |α| encountered: $(abs(stor.alpha))")
    end 
    adjoint_drift!(tmp, stor.alpha, algebra, system, stor)
    final_first_der = ForwardDiff.value(adjoint_drift_obj_1st_der(stor.alpha, costate, algebra, system, solver, stor))
    final_second_der = ForwardDiff.value(adjoint_drift_obj_2nd_der(stor.alpha, costate, algebra, system, solver, stor))
    @assert isapprox(final_first_der, 0.0, atol=1e-10) && final_second_der < 0 "Maximisation of adjoint drift failed: f' = $final_first_der, f'' = $final_second_der"
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing
end 

# optimal_adjoint_drift!(tmp::Matrix{ComplexF64}, costate::Matrix{ComplexF64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
# ) = optimal_adjoint_drift_optimiser!(tmp, costate, algebra, system, stor)

# optimal_adjoint_drift!(tmp::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
# costate::Matrix{<:Complex{<:ForwardDiff.Dual}}, 

optimal_adjoint_drift!(tmp, costate, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_analytic!(tmp, costate, algebra, system, solver, stor)