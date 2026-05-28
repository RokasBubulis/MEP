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

    ad!(stor.tmp_mat_lie, x_lie, f)
    res_arr .= real.(expv(-α, stor.tmp_mat_lie, y_lie))
    return nothing 
end 

function adjoint_drift_obj(α::Float64, costate_arr::AbstractVector{S}, im_control_arr::Vector{Float64}, neg_im_drift_arr::Vector{Float64}, structure_tensor::Array{Float64, 3}, stor::Storage) where S
    # tr(HM) = -sum_mu,nu c_mu*d_nu where H = sum_mu c_mu L_mu, M = sum_nu d_nu L_nu

    adjoint_drift_new!(stor.tmp_adj_drift_arr, α, im_control_arr, neg_im_drift_arr, structure_tensor, stor)
    return -dot(stor.tmp_adj_drift_arr, costate_arr)
end

function optimal_adjoint_drift_lie!(res_arr::Vector{T}, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where {T,S}
    
    x0 = [stor.alpha]
    obj(x) = -adjoint_drift_obj(x[], costate_arr, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra.structure_tensor, stor)
    res = Optim.optimize(obj, x0, Newton(linesearch = LineSearches.BackTracking()))
    stor.alpha = Optim.minimizer(res)[]
    adjoint_drift_new!(res_arr, stor.alpha, algebra.im_control_lie, algebra.neg_im_drift_lie, algebra.structure_tensor, stor)

    return nothing 
end 
