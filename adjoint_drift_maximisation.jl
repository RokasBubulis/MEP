using Optim, LineSearches, Plots

include("generators.jl")
include("checks.jl")

function br!(res, A, B, tmp)
    mul!(res, A, B)
    mul!(tmp, B, A)
    res .-= tmp
    return nothing
end 

function check_duals(x, name)
    # Flatten to a plain Vector regardless of scalar/array/matrix input
    xs = x isa AbstractArray ? vec(x) : [x]
    
    T = eltype(xs)
    ET = T <: Complex ? real(T) : T
    ET <: ForwardDiff.Dual || return

    # # Extract partials from real part of each element
    all_partials = mapreduce(
        el -> collect(ForwardDiff.partials(real(el))),
        vcat,
        xs
    )

    # if all(iszero, all_partials)
    #     @warn("All dual parts are zero at $name — gradient not flowing!")
    if any(abs.(all_partials) .> 1e2)
        throw("Very large dual parts at $name, max_val=$(maximum(abs.(all_partials)))")
    end
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

    if real(eltype(res)) <: ForwardDiff.Dual
        x_lie_coeffs = stor.campbell_array1_dual
        y_lie_coeffs = stor.campbell_array2_dual
        last_term_lie_coeffs = stor.campbell_array3_dual
        new_term_lie_coeffs = stor.campbell_array4_dual
        res_lie_coeffs = stor.campbell_array5_dual
    else
        x_lie_coeffs = stor.campbell_array1
        y_lie_coeffs = stor.campbell_array2
        last_term_lie_coeffs = stor.campbell_array3
        new_term_lie_coeffs = stor.campbell_array4
        res_lie_coeffs = stor.campbell_array5
    end

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


function adjoint_drift!(res::Matrix{TCostate}, α::TAlpha, algebra::Algebra, system::System, stor::Storage) where {TAlpha, TCostate}
    adjoint_action_by_campbell_structure_tensor!(res, -α * system.im_control, -system.im_drift, algebra, stor)
    return nothing 
end 

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
    return real(tr(stor.tmp_adjoint_drift_2nd_der_obj)) - solver.lambda * 2
end 

function differentiable_mod(α, system)
    # 1/sqrt(5) are the eigenvalues of diagonal L
    α_mod = system.period_im_control / 2π * angle(exp(2π * im * α / system.period_im_control))
    return α_mod 
end 

function optimal_adjoint_drift_analytic!(tmp::Matrix{TCostate}, costate::Matrix{TCostate}, algebra::Algebra, system::System, solver::SolverParams, stor::Storage) where TCostate

    stor.tmp_primal_costate .= primal(costate)
    α = primal(stor.alpha)

    for i in 1:solver.Newton_steps
        first_der = adjoint_drift_obj_1st_der(α, stor.tmp_primal_costate, algebra, system, solver, stor)
        second_der = adjoint_drift_obj_2nd_der(α, stor.tmp_primal_costate, algebra, system, solver, stor)
        dα = solver.Newton_damping * first_der / second_der
        abs(dα) < solver.Newton_tol && break
        # @show first_der
        # @show second_der
        # @show dα
        if second_der < 0
            α -= dα
        elseif second_der > 0
            α += dα
        end 
        
        α = differentiable_mod(α, system)

    end 

    if abs(α) > 8.0 # alpha=8 corresponds to an error of order -9
        @warn("Unusually large |α| encountered: $(abs(α))")
    end 

    if real(eltype(costate)) <: ForwardDiff.Dual 

        f1_dual = adjoint_drift_obj_1st_der(α, costate, algebra, system, solver, stor)
        f2_primal = adjoint_drift_obj_2nd_der(α, stor.tmp_primal_costate, algebra, system, solver, stor)

        # @show f2_primal
        if abs(f2_primal) > solver.tol 
            p = ForwardDiff.partials(real(f1_dual)) / (-f2_primal)
        else 
            p = ForwardDiff.Partials(ntuple(_ -> zero(Float64), ForwardDiff.npartials(real(f1_dual))))
        end 

        stor.alpha = ForwardDiff.Dual{ForwardDiff.tagtype(typeof(real(stor.alpha)))}(α, p)

    else 
        stor.alpha = α
    end 

    adjoint_drift!(tmp, stor.alpha, algebra, system, stor)

    final_first_der = adjoint_drift_obj_1st_der(primal(stor.alpha), stor.tmp_primal_costate, algebra, system, solver, stor)
    final_second_der = adjoint_drift_obj_2nd_der(primal(stor.alpha), stor.tmp_primal_costate, algebra, system, solver, stor)

    if ! isapprox(final_first_der, 0.0, atol=1e-10) && final_second_der < 0
        @warn "Maximisation of adjoint drift failed: f' = $final_first_der, f'' = $final_second_der, α = $(stor.alpha)"
    end
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