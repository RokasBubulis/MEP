"distance valid only if diagonal control"

function project_propagator_to_logical_subspace!(tmp, U)
    tmp[1:2, 1:2] .= U[1:2, 1:2]
    tmp[1:2, 3:4] .= U[1:2, 4:5]
    tmp[3:4, 1:2] .= U[4:5, 1:2]
    tmp[3:4, 3:4] .= U[4:5, 4:5]

    return nothing
end 

function project_from_3_to_2_levels!(res, U, algebra, stor)
    # U^(2) = Φ * Ψ' * U^(3) * Ψ * Φ' 
    # where Ψ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0,0), (0,1,0)}
    # Φ is a matrix stacked with basis vectors obtained by n tensor products among {(1,0), (0,1)}
    # where n is the number of particles 
    # this allows projection from 3 level operator to 2 where |r> states are excluded

    mul!(stor.tmp_logical1, U, algebra.ryd_to_logic_conv_mat)
    mul!(res, algebra.adj_ryd_to_logic_conv_mat, stor.tmp_logical1)
    return nothing
end

struct DistFunctor
    exp_buffer     ::Vector{ComplexF64}
    im_control_vec ::Vector{ComplexF64}
    tmp_diag       ::Vector{ComplexF64}
    n              ::Int
end

function (f::DistFunctor)(β)
    f.exp_buffer .= exp.(β .* f.im_control_vec)
    return 1 - (1/f.n) * abs(dot(f.exp_buffer, f.tmp_diag))
end

function distance(U::AbstractMatrix, tar::TargetContainer, algebra::Algebra, stor::Storage)

    n = size(tar.adjoint_target, 1)

    if size(U, 1) != n
        project_from_3_to_2_levels!(stor.U_2levels, U, algebra, stor)
        mul!(stor.tmp_2levels, stor.U_2levels, tar.adjoint_target)
        for i in eachindex(stor.tmp_diag_2levels)
            stor.tmp_diag_2levels[i] = stor.tmp_2levels[i, i]
        end
        tmp_diag       = stor.tmp_diag_2levels
        im_control_vec = algebra.im_control_vec_2levels
        exp_buffer     = stor.exp_buffer_dist_2levels
    else
        mul!(stor.tmp_3levels, U, tar.adjoint_target)
        for i in eachindex(stor.tmp_diag_3levels)
            stor.tmp_diag_3levels[i] = stor.tmp_3levels[i, i]
        end
        tmp_diag       = stor.tmp_diag_3levels
        im_control_vec = algebra.im_control_vec_3levels
        exp_buffer     = stor.exp_buffer_dist_3levels
    end

    dist = DistFunctor(exp_buffer, im_control_vec, tmp_diag, n)
    res   = optimize(dist, -2π, 2π, Brent())
    β_opt = Optim.minimizer(res)
    return dist(β_opt)
end