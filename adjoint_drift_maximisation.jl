using Optim, LineSearches

include("generators.jl")
include("checks.jl")

# depth of 10 is insufficient to produce oscillatory behaviour with equal peaks
# TODO test if depth of 20 is sufficient
function adjoint_action_by_campbell(X::SparseMatrixCSC{TX, Int}, 
    Y::SparseMatrixCSC{TY, Int}; depth = 20) where {TX, TY}
    # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
    # X = -α i control, Y = -i drift

    T = promote_type(TX, TY)
    result = T.(Y)
    last_term = T.(Y)
    coeff = one(T)
    for n in 1:depth
        new_term = br(X, last_term)
        coeff /= n
        result .+= coeff .* new_term
        last_term = new_term
    end
    result ./= norm(result)
    return result
end

# temporary expensive implementation, approach above should be made efficient to use later
# function adjoint_action_by_campbell(X::SparseMatrixCSC{TX, Int}, 
#     Y::SparseMatrixCSC{TY, Int}) where {TX, TY}
#     # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
#     # X = -α i control, Y = -i drift
#     return exp(Matrix(X)) * Y * exp(-Matrix(X))
# end

function adjoint_drift!(tmp::Matrix{TCostate}, neg_im_drift::SparseMatrixCSC{TSystem,Int}, 
    im_control::SparseMatrixCSC{TSystem,Int}, α::TAlpha) where {TAlpha, TSystem, TCostate}
    tmp .= adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return nothing 
end 

function adjoint_drift(neg_im_drift::SparseMatrixCSC{TSystem,Int},
    im_control::SparseMatrixCSC{TSystem,Int}, α::TAlpha) where {TAlpha, TSystem}
    mat = adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return mat
end 

# Negate the objective and derivatives as the goal is to maximise the function
function neg_adjoint_drift_obj(x::Vector{TAlpha}, neg_im_drift::SparseMatrixCSC{TSystem,Int},
    im_control::SparseMatrixCSC{TSystem,Int}, costate::Matrix{TCostate}) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    return -real(tr(control_H * costate))
end 

function neg_adjoint_drift_obj_1st_der!(G::AbstractVector{TAlpha}, x::AbstractVector{TAlpha}, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    G[1] = -real(tr(first_der * costate))
    return nothing
end 

function neg_adjoint_drift_obj_2nd_der!(H::Matrix{TAlpha}, x::AbstractVector{TAlpha}, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}) where {TAlpha, TSystem, TCostate}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    H[1,1] = -real(tr(second_der * costate))
    return nothing 
end

function optimal_adjoint_drift_optimiser!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, params::Params) where TCostate
    neg_im_drift = -params.physics.im_drift
    im_control = params.physics.im_control
    x0 = [0.0]

    td = TwiceDifferentiable(
    x -> neg_adjoint_drift_obj(x, neg_im_drift, im_control, costate),
    (G, x) -> neg_adjoint_drift_obj_1st_der!(G, x, neg_im_drift, im_control, costate),
    (H, x) -> neg_adjoint_drift_obj_2nd_der!(H, x, neg_im_drift, im_control, costate),
    x0
    )
    res = Optim.optimize(td, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[1]

    adjoint_drift!(tmp, neg_im_drift, im_control, α_optimal)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing
end 

### Analytic method to optimisation required to enable forward differentiation


function neg_adjoint_drift_obj_1st_der(α::TAlpha, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}) where {TAlpha, TSystem, TCostate}

    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    return -real(tr(first_der * costate))
end 

function neg_adjoint_drift_obj_2nd_der(α::TAlpha, 
    neg_im_drift::SparseMatrixCSC{TSystem,Int}, im_control::SparseMatrixCSC{TSystem,Int}, 
    costate::Matrix{TCostate}) where {TAlpha, TSystem, TCostate}

    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    return -real(tr(second_der * costate))
end

function optimal_adjoint_drift_analytic!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, params::Params) where TCostate
    neg_im_drift = -params.physics.im_drift
    im_control = params.physics.im_control
    α = zero(real(TCostate))

    for _ in 1:params.solver.Newton_steps
        first_der = neg_adjoint_drift_obj_1st_der(α, neg_im_drift, im_control, costate)
        second_der = neg_adjoint_drift_obj_2nd_der(α, neg_im_drift, im_control, costate)
        dα = first_der / second_der
        α -= first_der / second_der
        abs(dα) < 1e-10 && break
    end 

    adjoint_drift!(tmp, neg_im_drift, im_control, α)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(tmp)

    return nothing
end 