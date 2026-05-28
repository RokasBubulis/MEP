using Optim, LineSearches, Plots

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

function adjoint_action_by_campbell_structure_tensor!(res, X, Y, algebra::Algebra, stor::Storage; depth = 20)

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
    return nothing

end 

function  adjoint_action_true(X, Y)
    return exp(Matrix(X)) * Y * exp(-Matrix(X))
end

function  adjoint_action_true!(res, X, Y, stor)
    copyto!(stor.tmp1, X)
    LinearAlgebra.exp!(stor.tmp1) 
    mul!(stor.tmp2, stor.tmp1, Y)
    adjoint!(stor.tmp1_adj, stor.tmp1)
    mul!(res, stor.tmp2, stor.tmp1_adj)
    return nothing
end

function adjoint_drift!(res::Matrix{ComplexF64}, α::Float64, algebra::Algebra, system::System, stor::Storage)
    copyto!(stor.tmp, system.im_control)
    lmul!(-α, stor.tmp) 
    adjoint_action_true!(res, stor.tmp, -system.im_drift, stor)
    return nothing 
end 

# function adjoint_drift!(res::Matrix{<:Complex{<:ForwardDiff.Dual}}, α::TAlpha, algebra::Algebra, system::System, stor::Storage) where TAlpha
#     copyto!(stor.tmp_dual, system.im_control)
#     lmul!(-α, stor.tmp_dual) 
#     adjoint_action_by_campbell_structure_tensor!(res, stor.tmp_dual, -system.im_drift, algebra, stor)
#     return nothing 
# end 


function adjoint_drift_obj(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}
    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    mul!(stor.tmp_adjoint_drift_obj, stor.tmp_adjoint_drift, costate)
    return real(tr(stor.tmp_adjoint_drift_obj))
end

function adjoint_drift_obj_1st_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for first: ")
    if real(eltype(costate)) <: ForwardDiff.Dual 
        res = stor.tmp_adjoint_drift_1st_der_obj_dual
    else
        res = stor.tmp_adjoint_drift_1st_der_obj
    end 
    mul!(res, stor.tmp_adjoint_drift_1st_der, costate)
    return real(tr(res))
end 

function adjoint_drift_obj_2nd_der(α::TAlpha, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where {TAlpha, TCostate}

    adjoint_drift!(stor.tmp_adjoint_drift, α, algebra, system, stor)
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_1st_der, stor.tmp_adjoint_drift, system.im_control, algebra, stor; identifier="First der for second: ")
    bracket_via_lie_coeffs!(stor.tmp_adjoint_drift_2nd_der, stor.tmp_adjoint_drift_1st_der, system.im_control, algebra, stor; identifier="Second der: ")
    mul!(stor.tmp_adjoint_drift_2nd_der_obj, stor.tmp_adjoint_drift_2nd_der, costate)
    return real(tr(stor.tmp_adjoint_drift_2nd_der_obj))
end 


function optimal_adjoint_drift_optimiser!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where TCostate
    x0 = [0.0]

    td = TwiceDifferentiable(
    x -> -adjoint_drift_obj(x[], costate, algebra, solver, stor),
    (G, x) -> (G .= -adjoint_drift_obj_1st_der(x[], costate, algebra, system, solver, stor)),
    (H, x) -> (H .= -adjoint_drift_obj_2nd_der(x[], costate, algebra, system, solver, stor)),
    x0
    )
    res = Optim.optimize(td, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[]
    adjoint_drift!(tmp, α_optimal , algebra, system, stor)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing 
end 


optimal_adjoint_drift!(tmp::Matrix{ComplexF64}, costate::Matrix{ComplexF64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_optimiser!(tmp, costate, algebra, system, solver, stor)