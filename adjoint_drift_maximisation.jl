using Optim, LineSearches

include("generators.jl")
include("checks.jl")

# depth of 10 is insufficient to produce oscillatory behaviour with equal peaks
# function adjoint_action_by_campbell(X::SparseMatrixCSC{T, Int}, 
#     Y::SparseMatrixCSC{T, Int}; depth = 10)
#     # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
#     # X = -α i control, Y = -i drift

#     result = copy(Y)
#     last_term = copy(Y)
#     coeff = 1.0
#     for n in 1:depth
#         new_term = br(X, last_term)
#         coeff /= n
#         result .+= coeff .* new_term
#         last_term = new_term
#     end
#     result ./= norm(result)
#     return result
# end

# temporary expensive implementation, approach above should be made efficient to use later
function adjoint_action_by_campbell(X::SparseMatrixCSC{TC, Int}, 
    Y::SparseMatrixCSC{TC, Int}; depth = 10) where TC
    # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
    # X = -α i control, Y = -i drift
    return exp(Matrix(X)) * Y * exp(-Matrix(X))
end

function adjoint_drift!(tmp::Matrix{TC}, neg_im_drift::SparseMatrixCSC{TC,Int}, 
    im_control::SparseMatrixCSC{TC,Int}, α::TR) where {TR, TC}
    tmp .= adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return nothing 
end 

function adjoint_drift(neg_im_drift::SparseMatrixCSC{TC,Int},
    im_control::SparseMatrixCSC{TC,Int}, α::TR) where {TR, TC}
    mat = adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return mat
end 

# Negate the objective and derivatives as the goal is to maximise the function
function neg_adjoint_drift_obj(x::Vector{TR}, neg_im_drift::SparseMatrixCSC{TC,Int},
    im_control::SparseMatrixCSC{TC,Int}, costate::Matrix{TC}) where {TR, TC}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    return -real(tr(control_H * costate))
end 

function neg_adjoint_drift_obj_1st_der!(G::AbstractVector{TR}, x::AbstractVector{TR}, 
    neg_im_drift::SparseMatrixCSC{TC,Int}, im_control::SparseMatrixCSC{TC,Int}, 
    costate::Matrix{TC}) where {TR, TC}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    G[1] = -real(tr(first_der * costate))
    return nothing
end 

function neg_adjoint_drift_obj_2nd_der!(H::Matrix{TR}, x::AbstractVector{TR}, 
    neg_im_drift::SparseMatrixCSC{TC,Int}, im_control::SparseMatrixCSC{TC,Int}, 
    costate::Matrix{TC}) where {TR, TC}
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    H[1,1] = -real(tr(second_der * costate))
    return nothing 
end

function optimal_adjoint_drift_optimiser!(tmp::Matrix{TC}, costate::Matrix{TC}, params::Params) where TC
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
