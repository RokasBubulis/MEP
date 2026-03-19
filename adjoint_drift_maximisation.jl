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
function adjoint_action_by_campbell(X::SparseMatrixCSC{T, Int}, 
    Y::SparseMatrixCSC{T, Int}; depth = 10)
    # e^X Y e^(-X) = \sum_n=0^inf 1/n! [X,Y]_n
    # X = -α i control, Y = -i drift
    return exp(Matrix(X)) * Y * exp(-Matrix(X))
end

function adjoint_drift!(tmp, neg_im_drift, im_control, α)
    tmp .= adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return nothing 
end 

function adjoint_drift(neg_im_drift, im_control, α)
    mat = adjoint_action_by_campbell(-α * im_control, neg_im_drift)
    return mat
end 

# Negate the objective and derivatives as the goal is to maximise the function
function neg_adjoint_drift_obj(x, neg_im_drift, im_control, costate)
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    return -real(tr(control_H * costate))
end 

function neg_adjoint_drift_obj_1st_der!(G, x, neg_im_drift, im_control, costate)
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    G[1] = -real(tr(first_der * costate))
    return nothing
end 

function neg_adjoint_drift_obj_2nd_der!(H, x, neg_im_drift, im_control, costate)
    α = x[1]
    control_H = adjoint_drift(neg_im_drift, im_control, α)
    first_der = control_H * im_control - im_control * control_H
    second_der =  first_der * im_control - im_control * first_der
    H[1,1] = -real(tr(second_der * costate))
    return nothing 
end

function optimal_adjoint_drift_newton!(costate, params)
    neg_im_drift = -params.system_params.im_drift
    im_control = params.system_params.im_control
    x0 = [0.0]

    td = TwiceDifferentiable(
    x -> neg_adjoint_drift_obj(x, neg_im_drift, im_control, costate),
    (G, x) -> neg_adjoint_drift_obj_1st_der!(G, x, neg_im_drift, im_control, costate),
    (H, x) -> neg_adjoint_drift_obj_2nd_der!(H, x, neg_im_drift, im_control, costate),
    x0
    )
    res = Optim.optimize(td, x0, Newton(linesearch = LineSearches.BackTracking()))
    α_optimal = Optim.minimizer(res)[1]

    adjoint_drift!(params.storage_params.adjoint_drift_tmp, neg_im_drift, im_control, α_optimal)
    # ensure optimal adjoint drift is anti-hermitian
    check_anti_hermiticity(params.storage_params.adjoint_drift_tmp)

    return nothing
end 

# function optimal_adjoint_drift_fminbox!(costate, params)

#     drift, control = params.drift, params.control
#     x0 = [0.0]
#     lower = [params.min_alpha]
#     upper = [params.max_alpha]

#     od = OnceDifferentiable(
#         x -> neg_adjoint_drift_obj(x, drift, control, costate),
#         (G, x) -> neg_adjoint_drift_obj_1st_der!(G, x, drift, control, costate),
#         x0
#     )
#     res = Optim.optimize(od, lower, upper, x0, Fminbox(BFGS()))
#     α_optimal = Optim.minimizer(res)[1]

#     adjoint_drift!(params.H_alpha_tmp, drift, control, α_optimal)

#     return nothing 
# end 

# function optimal_adjoint_drift_ipnewton!(costate, params)

#     drift, control = params.drift, params.control
#     x0 = [0.0]

#     td = TwiceDifferentiable(
#     x -> neg_adjoint_drift_obj(x, drift, control, costate),
#     (G, x) -> neg_adjoint_drift_obj_1st_der!(G, x, drift, control, costate),
#     (H, x) -> neg_adjoint_drift_obj_2nd_der!(H, x, drift, control, costate),
#     x0
#     )
#     constraints = TwiceDifferentiableConstraints([params.min_alpha], [params.max_alpha])
#     res = Optim.optimize(td, constraints, x0, IPNewton())

#     α_optimal = Optim.minimizer(res)[1]

#     adjoint_drift!(params.H_alpha_tmp, drift, control, α_optimal)

#     return nothing 
# end 

