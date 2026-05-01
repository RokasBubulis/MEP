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

function lie_bracket_coeffs!(res, f, x::Vector, y::Vector)
    n = length(x)
    for c in 1:n
        res[c] = sum(f[c,a,b] * x[a] * y[b] for a in 1:n, b in 1:n)
    end 
    return nothing 
end 

function adjoint_action_by_campbell_structure_tensor!(res, X::SparseMatrixCSC{TX, Int}, 
    Y::SparseMatrixCSC{TY, Int}, algebra::Algebra, stor::Storage; depth = 20) where {TX, TY}

    x_lie_coeffs = stor.tmp_array1
    y_lie_coeffs = stor.tmp_array2
    last_term_lie_coeffs = stor.tmp_array3
    new_term_lie_coeffs = stor.tmp_array4
    res_lie_coeffs = stor.tmp_array5

    project_to_algebra!(x_lie_coeffs, X, algebra, stor)
    project_to_algebra!(y_lie_coeffs, Y, algebra, stor)

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

function adjoint_drift!(tmp::Matrix{TCostate}, neg_im_drift::SparseMatrixCSC{TSystem,Int}, 
    im_control::SparseMatrixCSC{TSystem,Int}, α::TAlpha, algebra::Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}
    # adjoint_action_by_campbell!(tmp, -α * im_control, neg_im_drift, stor)
    adjoint_action_by_campbell_structure_tensor!(tmp, -α * im_control, neg_im_drift, algebra, stor)
    return nothing 
end 

function adjoint_drift(neg_im_drift::SparseMatrixCSC{TSystem,Int},
    im_control::SparseMatrixCSC{TSystem,Int}, α::TAlpha, algebra::Algebra, stor::Storage) where {TAlpha, TSystem}
    # adjoint_action_by_campbell!(stor.tmp, -α * im_control, neg_im_drift, stor)
    adjoint_action_by_campbell_structure_tensor!(stor.tmp, -α * im_control, neg_im_drift, algebra, stor)
    return stor.tmp
end 

# Negate the objective and derivatives as the goal is to maximise the function
function neg_adjoint_drift_obj(x::Vector{TAlpha}, neg_im_drift::SparseMatrixCSC{TSystem,Int},
    im_control::SparseMatrixCSC{TSystem,Int}, costate::Matrix{TCostate}, Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α, algebra, stor)
    return -real(tr(control_H * costate))
end 

function neg_adjoint_drift_obj_1st_der!(G::AbstractVector{TAlpha}, x::AbstractVector{TAlpha}, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}, algebra::Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α, algebra, stor)
    first_der = control_H * im_control - im_control * control_H
    G[1] = -real(tr(first_der * costate))
    return nothing
end 

function neg_adjoint_drift_obj_2nd_der!(H::Matrix{TAlpha}, x::AbstractVector{TAlpha}, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}, algebra::Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α, algebra, stor)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    H[1,1] = -real(tr(second_der * costate))
    return nothing 
end

function optimal_adjoint_drift_optimiser!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, algebra::Algebra, system::System, stor::Storage) where TCostate
    neg_im_drift = -system.im_drift
    im_control = system.im_control
    x0 = [0.0]

    td = TwiceDifferentiable(
    x -> neg_adjoint_drift_obj(x, neg_im_drift, im_control, costate, algebra, stor),
    (G, x) -> neg_adjoint_drift_obj_1st_der!(G, x, neg_im_drift, im_control, costate, algebra, stor),
    (H, x) -> neg_adjoint_drift_obj_2nd_der!(H, x, neg_im_drift, im_control, costate, algebra, stor),
    x0
    )
    res = Optim.optimize(td, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[1]

    adjoint_drift!(tmp, neg_im_drift, im_control, α_optimal, algebra, stor)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing
end 

### Analytic method to optimisation required to enable forward differentiation


function neg_adjoint_drift_obj_1st_der(α::TAlpha, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}, algebra::Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}

    control_H = adjoint_drift(neg_im_drift, im_control, α, algebra, stor)
    first_der = control_H * im_control - im_control * control_H
    return -real(tr(first_der * costate))
end 

function neg_adjoint_drift_obj_2nd_der(α::TAlpha, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}, algebra::Algebra, stor::Storage) where {TAlpha, TSystem, TCostate}

    control_H = adjoint_drift(neg_im_drift, im_control, α, algebra, stor)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    return -real(tr(second_der * costate))
end

function optimal_adjoint_drift_analytic!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where TCostate
    neg_im_drift = -system.im_drift
    im_control = system.im_control
    α = zero(real(TCostate))

    for _ in 1:solver.Newton_steps
        first_der = neg_adjoint_drift_obj_1st_der(α, neg_im_drift, im_control, costate, algebra, stor)
        second_der = neg_adjoint_drift_obj_2nd_der(α, neg_im_drift, im_control, costate, algebra, stor)
        dα = first_der / second_der
        α -= first_der / second_der
        abs(dα) < solver.Newton_tol && break
    end 
    if abs(ForwardDiff.value(α)) > 10.0
        @warn "Unusually large |α| encountered: $(abs(ForwardDiff.value(α)))"
    end 
    adjoint_drift!(tmp, neg_im_drift, im_control, α, algebra, stor)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing
end 

optimal_adjoint_drift!(tmp::Matrix{ComplexF64}, costate::Matrix{ComplexF64}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_optimiser!(tmp, costate, algebra, system, stor)

optimal_adjoint_drift!(tmp::Matrix{<:Complex{<:ForwardDiff.Dual}}, 
costate::Matrix{<:Complex{<:ForwardDiff.Dual}}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage
) = optimal_adjoint_drift_analytic!(tmp, costate, algebra, system, solver, stor)