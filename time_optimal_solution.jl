using Optim, FastExpm
include("lie_algebra.jl")

function obtain_target_kak_decomposition(target, drift, controls)

    g_c, l_c, p_c, a_c = construct_algebras(drift, controls)

    println("len(g_c) = $(length(g_c))")
    # for g in g_c
    #     display(g)
    # end

    println("len(l_c) = $(length(l_c))")
    # for l in l_c
    #     display(l)
    # end
    println("len(p_c) = $(length(p_c))")
    # for p in p_c
    #     display(p)
    # end
    println("len(a_c) = $(length(a_c))")
    # control_subgroup_basis = construct_subgroup_basis(l_c, 10)
    # println("len(l_c subgroup) = $(length(control_subgroup_basis))")
    # full_subgroup_basis = construct_subgroup_basis(g_c, 10)
    # println("len(g_c subgroup) = $(length(full_subgroup_basis))")


    function cost(v)
        dim_lc = length(l_c)
        dim_ac = length(a_c)
        x = v[1:dim_lc]
        y = v[(dim_lc + 1) : (dim_lc + dim_ac)]
        z = v[(dim_lc + 1 + dim_ac) : (2*dim_lc + dim_ac)]

        U = fastExpm(sum(x[i] * l_c[i] for i in 1:dim_lc))
        expDelta = fastExpm(sum(y[i] * a_c[i] for i in 1:dim_ac))
        V = fastExpm(sum(z[i] * l_c[i] for i in 1:dim_lc))
        M = U * expDelta * V

        return norm(target - M)
    end

    v0 = ones(2*length(l_c) + length(a_c))
    result = optimize(cost, v0, NelderMead())#, Optim.Options(show_trace=true))
    # println("Optimized x: ", Optim.minimizer(result))
    println("Minimum value: ", Optim.minimum(result))
    println("Converged: ", Optim.converged(result))
end

# control_subgroup_basis = construct_subgroup_basis(l_c, product_depth)
# println("len(l_c subgroup) = $(length(control_subgroup_basis))")
# weyl_group_orbit = construct_weyl_group_orbit(control_subgroup_basis, a_c)
# println("len(W orbit) = $(length(weyl_group_orbit))")
