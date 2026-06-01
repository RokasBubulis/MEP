"distance valid only if diagonal control"

function project_propagator_to_logical_subspace!(tmp, U)
    tmp[1:2, 1:2] .= U[1:2, 1:2]
    tmp[1:2, 3:4] .= U[1:2, 4:5]
    tmp[3:4, 1:2] .= U[4:5, 1:2]
    tmp[3:4, 3:4] .= U[4:5, 4:5]

    return nothing
end 

function project_from_3_to_2_levels!(res, U, system, stor)
    # U^(2) = Φ * Ψ' * U^(3) * Ψ * Φ' 
    # where Ψ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0,0), (0,1,0)}
    # Φ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0), (0,1)}
    # where n is the number of particles 
    # this allows projection from 3 level operator to 2 where |r> states are excluded

    mul!(stor.tmp_logical1, U, system.ryd_to_logic_conv_mat)
    mul!(res, system.ryd_to_logic_conv_mat_adj, stor.tmp_logical1)
    return nothing
end

function distance_objective_optimiser(U::AbstractMatrix, system::System, stor::Storage)

    # project_propagator_to_logical_subspace!(stor.U_logic, U)
    project_from_3_to_2_levels!(stor.U_logic, U, system, stor)
    mul!(stor.tmp_logic, stor.U_logic, system.adjoint_target_logic)
    tmp_diag = diag(stor.tmp_logic)
    dist(β) = 1 - 1/size(stor.U_logic,1) * abs(dot(exp.(β.*system.im_control_vec_logic), tmp_diag))
    # res = optimize(dist, -pi, pi)
    res = optimize(dist, -2π, 2π, Brent(), 
    # abs_tol = 1e-12,
    # rel_tol = 1e-12
)   
    β_opt = Optim.minimizer(res)
    dist_opt = dist(β_opt)
    # @assert dist_opt >= 0.0 "Invalid distance obtained: dist = $(dist_opt), beta=$β_opt, exp_vec = $(exp.(β_opt.*system.im_control_vec_logic)), tmp_diag=$tmp_diag, dot=$(dot(exp.(β_opt.*system.im_control_vec_logic), tmp_diag)))"
    return dist_opt
end

distance(U::AbstractMatrix{<:Complex}, system::System, solver::SolverParams, stor::Storage) = distance_objective_optimiser(U, system, stor)