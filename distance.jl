"distance calculations via optimiser and analytical are valid only if diagonal control"

function distance_objective_optimiser(U::Union{Matrix{T}, SparseMatrixCSC{T,Int}}, system::System, stor::Storage) where T

    mul!(stor.tmp, U, system.adjoint_target)
    tmp_diag = diag(stor.tmp)

    dist(β) = 1 - 1/size(U,1) * abs(dot(exp.(β.*system.im_control_vec), tmp_diag))
    # res = optimize(dist, -pi, pi)
    res = optimize(dist, -π, π, Brent(), 
    abs_tol = 1e-12,   # tighter argument tolerance
    rel_tol = 1e-12
)
    β_opt = Optim.minimizer(res)

    return dist(β_opt)
end

function distance_objective_analytic(β::TBeta, U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    system::System, stor::Storage) where {TBeta, TCostate, TSystem}

    mul!(stor.tmp, U, system.adjoint_target)
    A_diag = diag(stor.tmp) # A_jj 
    dim = size(stor.tmp, 1)
    expLBeta = exp.(β.*system.im_control_vec)
    d = dot(expLBeta, A_diag)
    res =  1 - 1/dim * real(d)
    return res
end

function distance_objective_analytic_derivatives(β::TBeta, U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    system::System, stor::Storage) where {TBeta, TCostate, TSystem}

    first_der, second_der = zero(TBeta), zero(TBeta)
    dim = size(stor.tmp, 1)
    mul!(stor.tmp, U, system.adjoint_target)
    A_diag = diag(stor.tmp) # A_jj 
    LexpBetaL = system.im_control_vec .* exp.(β.*system.im_control_vec)
    first_der = -1/dim * real(dot(LexpBetaL, A_diag))
    L2expBetaL = system.im_control_vec .* LexpBetaL
    second_der = -1/dim * real(dot(L2expBetaL, A_diag))

    return first_der, second_der
end 

function minimum_distance_objective_analytic(U::Union{Matrix{TCostate}, SparseMatrixCSC{TSystem, Int}},
    system::System, solver::SolverParams, stor::Storage) where {TCostate, TSystem}

    β = zero(real(eltype(U)))
    #beta = primal(β)
    for _ in 1:solver.Newton_steps
        first_der, second_der = distance_objective_analytic_derivatives(β, U, system, stor)
        dβ = first_der / second_der
        β -= dβ
        abs(dβ) < solver.Newton_tol && break
    end 

    beta = ForwardDiff.value(β)
    final_first_der, final_second_der = distance_objective_analytic_derivatives(beta, U, system, stor)

    if real(eltype(U)) <: ForwardDiff.Dual 
        p = ForwardDiff.partials(real(final_first_der)) / (-ForwardDiff.value(final_second_der))
        beta_trial = ForwardDiff.Dual{ForwardDiff.tagtype(typeof(real(β)))}(beta, p)
        @assert isapprox(β, beta_trial) "β: $β, reconstructed: $beta_trial"
    end 

    @assert isapprox(final_first_der, 0.0, atol=1e-10) && final_second_der > 0 "Minimisation of distance to target coset failed: f' = $final_first_der, f'' = $final_second_der"
    min_dist = distance_objective_analytic(β, U, system, stor)
    return min_dist
end     

distance(U::Matrix{<:Complex{<:ForwardDiff.Dual}}, system::System, solver::SolverParams, 
stor::Storage
) = minimum_distance_objective_analytic(U, system, solver, stor)

distance(U::Union{Matrix{ComplexF64}, SparseMatrixCSC{ComplexF64, Int}}, system::System, solver::SolverParams, stor::Storage) = distance_objective_optimiser(U, system, stor)