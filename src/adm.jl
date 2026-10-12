using ExponentialUtilities, Optim
# using LineSearches   # only needed by the removed legacy Optim-based optimisers

# ------------------------------------------------------------------
# Core (used by the forward propagation path in test_forward.jl)
# ------------------------------------------------------------------

function adjoint_drift_efficient!(res_arr, α, KrylovSubspace, stor)
    expv!(res_arr, -α, KrylovSubspace; cache=stor.expv_cache)
    return nothing
end

function adjoint_drift_obj_derivatives(α::Float64, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where S
    # returns ( g , h ) = ( ∂h/∂α , ∂²h/∂α² ) evaluated at the current α
    #   h   = tr(H M)  = -∑ h_μ m_μ   (H = rotated drift)
    #   H'  = [H, i·∑ Z]   -> first-derivative coefficients
    #   H'' = [H', i·∑ Z]  -> second-derivative coefficients

    adjoint_drift_efficient!(stor.tmp_adj_drift_arr, α, algebra.KrylovSubspace, stor)
    lie_bracket_coeffs!(stor.tmp_adj_drift_first_der_arr, algebra.structure_tensor, stor.tmp_adj_drift_arr, algebra.im_control_lie)
    lie_bracket_coeffs!(stor.tmp_adj_drift_second_der_arr, algebra.structure_tensor, stor.tmp_adj_drift_first_der_arr, algebra.im_control_lie)
    return -dot(stor.tmp_adj_drift_first_der_arr, costate_arr), -dot(stor.tmp_adj_drift_second_der_arr, costate_arr)
end 

function update_alpha_newton!(stor, costate_arr, algebra::Algebra; gtol::Float64 = 1e-8, xtol::Float64 = 1e-8, maxiter::Int = 30, damping::Float64 = 1.0, curvature_tol::Float64 = 1e-12)

    α = stor.alpha
    for i in 1:maxiter
        g, h = adjoint_drift_obj_derivatives(α, costate_arr, algebra, stor)
        # convergence
        if abs(g) < gtol
            break
        end
        if abs(h) < curvature_tol
            break
        end

        step = g / h
        if abs(step) < xtol
            break
        end
        if h > 0 
            α += damping * step 
        else
            α -= damping * step
        end 
    end

    g_fin, h_fin = adjoint_drift_obj_derivatives(α, costate_arr, algebra, stor)
    if h_fin > curvature_tol 
        @warn("Positive final curvature for ADM. g: $g_fin, h: $h_fin")
    end 
    stor.alpha = α
    return nothing
end

function optimal_adjoint_drift_lie_analytic!(res_arr::Vector{T}, costate_arr::AbstractVector{S}, algebra::Algebra, stor::Storage) where {T,S}
    
    update_alpha_newton!(stor, costate_arr, algebra)
    adjoint_drift_efficient!(res_arr, stor.alpha, algebra.KrylovSubspace, stor)
    return nothing 
end 
