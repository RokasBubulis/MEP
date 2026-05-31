using ExponentialUtilities, Optim, LineSearches

function ad!(res_mat, x_lie, f)
    # ad_X(Y) = [X,Y], X,Y in Lie algebra

    res_mat .= 0
    for ν in eachindex(x_lie)
        for μ in ν+1:length(x_lie)
            xμ = x_lie[μ]
            xν = x_lie[ν]
            for λ in eachindex(x_lie)
                val = f[μ, ν, λ]
                res_mat[λ, ν] += xμ * val
                res_mat[λ, μ] -= xν * val
            end
        end
    end
    return nothing
end

function adjoint_drift_new!(res_arr, α, x_lie, y_lie, f, stor)
    # e^X Y e^-X, X=-α*im_control, Y = -im_drift

    ad!(stor.tmp_mat_lie, x_lie, f)   # TODO THIS SHOULD BE PRECOMPUTED AND STORED IN ALGEBRA INSTEAD!
    res_arr .= real.(expv(-α, stor.tmp_mat_lie, y_lie))
    return nothing 
end 

function adjoint_drift_efficient!(res_arr, α, y_lie, adj_repr_map)
    # e^X Y e^-X, X=-α*im_control, Y = -im_drift

    res_arr .= real.(expv(-α, adj_repr_map, y_lie))
    return nothing 
end 

function adjoint_drift_obj(α::Float64, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where S
    # tr(HM) = -sum_μν h_μ*m_ν where H = sum_μ h_μ L_μ, M = sum_ν m_ν L_ν

    adjoint_drift_efficient!(stor.tmp_adj_drift_arr, α, algebra.neg_im_drift_lie, algebra.adj_repr_map)
    return -dot(stor.tmp_adj_drift_arr, costate_arr)
end

function adjoint_drift_obj_1st_der(α::Float64, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where S
    # d/dα tr(HM) = tr(H'M), H' = [H, isum_j Z_j] = d_γ = ∑_μν h_μ*c_ν*f_μν^γ
    
    adjoint_drift_efficient!(stor.tmp_adj_drift_arr, α, algebra.neg_im_drift_lie, algebra.adj_repr_map)
    lie_bracket_coeffs!(stor.tmp_adj_drift_first_der_arr, algebra.structure_tensor, stor.tmp_adj_drift_arr, algebra.im_control_lie)
    return -dot(stor.tmp_adj_drift_first_der_arr, costate_arr)
end 

function adjoint_drift_obj_2nd_der(α::Float64, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where S
    # d^2/dα^2 tr(HM) = tr(H''M), H'' = [H', isum_j Z_j]
    
    adjoint_drift_efficient!(stor.tmp_adj_drift_arr, α, algebra.neg_im_drift_lie, algebra.adj_repr_map)
    lie_bracket_coeffs!(stor.tmp_adj_drift_first_der_arr, algebra.structure_tensor, stor.tmp_adj_drift_arr, algebra.im_control_lie)
    lie_bracket_coeffs!(stor.tmp_adj_drift_second_der_arr, algebra.structure_tensor, stor.tmp_adj_drift_first_der_arr, algebra.im_control_lie)
    return -dot(stor.tmp_adj_drift_second_der_arr, costate_arr)
end 

function optimal_adjoint_drift_lie!(res_arr::Vector{T}, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where {T,S}
    
    x0 = [stor.alpha]
    td = TwiceDifferentiable(
    x -> -adjoint_drift_obj(x[], costate_arr, algebra, stor),
    (G, x) -> (G .= -adjoint_drift_obj_1st_der(x[], costate_arr, algebra, stor)),
    (H, x) -> (H .= -adjoint_drift_obj_2nd_der(x[], costate_arr, algebra, stor)),
    x0
    )
    res = Optim.optimize(td, x0, Newton(linesearch = LineSearches.BackTracking()))

    stor.alpha = Optim.minimizer(res)[]
    adjoint_drift_efficient!(res_arr, stor.alpha, algebra.neg_im_drift_lie, algebra.adj_repr_map)

    return nothing 
end 

function optimal_adjoint_drift_lie_nondiff!(res_arr::Vector{T}, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where {T,S}
    
    x0 = [stor.alpha]

    obj(x) = -adjoint_drift_obj(x[], costate_arr, algebra, stor)
    res = Optim.optimize(obj, x0, Newton(linesearch = LineSearches.BackTracking()))
    stor.alpha = Optim.minimizer(res)[]
    adjoint_drift_efficient!(res_arr, stor.alpha, algebra.neg_im_drift_lie, algebra.adj_repr_map)

    return nothing 
end 
