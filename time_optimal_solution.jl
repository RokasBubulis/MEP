using Optim, FastExpm
include("lie_algebra.jl")

function obtain_target_kak_decomposition(target, drift, controls)

    g_c, l_c, p_c, a_c = construct_algebras(drift, controls)
    println("len(g_c) = $(length(g_c))")
    println("len(l_c) = $(length(l_c))")
    println("len(p_c) = $(length(p_c))")
    println("len(a_c) = $(length(a_c))")
    dim_lc = length(l_c)
    dim_ac = length(a_c)
    x_array = 1:dim_lc
    y_array = (dim_lc + 1) : (dim_lc + dim_ac)
    z_array = (dim_lc + 1 + dim_ac) : (2*dim_lc + dim_ac)

    function cost(v)
        x = v[x_array]
        y = v[y_array]
        z = v[z_array]

        U = fastExpm(sum(x[i] * l_c[i] for i in 1:dim_lc))
        expDelta = fastExpm(sum(y[i] * a_c[i] for i in 1:dim_ac))
        V = fastExpm(sum(z[i] * l_c[i] for i in 1:dim_lc))
        M = U * expDelta * V

        return norm(target - M)
    end

    v0 = ones(2*length(l_c) + length(a_c))
    result = optimize(cost, v0, NelderMead(), Optim.Options(
        iterations = 50_000
    ))
    #, Optim.Options(show_trace=true))
    coeffs = Optim.minimizer(result)
    x, y, z = coeffs[x_array], coeffs[y_array], coeffs[z_array]
    println("Error: ", Optim.minimum(result))
    println("Converged: ", Optim.converged(result))

    # U = fastExpm(sum(x[i] * l_c[i] for i in 1:dim_lc))
    # Delta0 = sum(y[i] * a_c[i] for i in 1:dim_ac)
    # V = fastExpm(sum(z[i] * l_c[i] for i in 1:dim_lc))
    # M = U*fastExpm(Delta0)*V
    # display(sparse(round.(M, digits=6)))
end

# control_subgroup_basis = construct_subgroup_basis(l_c, product_depth)
# println("len(l_c subgroup) = $(length(control_subgroup_basis))")
# weyl_group_orbit = construct_weyl_group_orbit(control_subgroup_basis, a_c)
# println("len(W orbit) = $(length(weyl_group_orbit))")


function obtain_target_kak_decomposition_single_control(target, gens)
    g_c, l_c, p_c, a_c = construct_algebras_single_control(gens)
    println("len(g_c) = $(length(g_c))")
    println("len(l_c) = $(length(l_c))")
    println("len(p_c) = $(length(p_c))")
    println("len(a_c) = $(length(a_c))")
    dim_ac = length(a_c)
    y_array = 3 : (2 + dim_ac)

    function cost(v)
        x = v[1]
        y = v[y_array]
        z = v[2]

        U = fastExpm(x * l_c[1])
        expDelta = fastExpm(sum(y[i] * a_c[i] for i in 1:dim_ac))
        V = fastExpm(z * l_c[1])
        M = U * expDelta * V

        return norm(target - M)
    end

    v0 = ones(2*length(l_c) + length(a_c))
    result = optimize(cost, v0, NelderMead(), Optim.Options(
        iterations = 50_000))
    #, Optim.Options(show_trace=true))
    coeffs = Optim.minimizer(result)
    # x, y, z = coeffs[x_array], coeffs[y_array], coeffs[z_array]
    println("Error: ", Optim.minimum(result))
    println("Converged: ", Optim.converged(result))

    # U = fastExpm(sum(x[i] * l_c[i] for i in 1:dim_lc))
    # Delta0 = sum(y[i] * a_c[i] for i in 1:dim_ac)
    # V = fastExpm(sum(z[i] * l_c[i] for i in 1:dim_lc))
    # M = U*fastExpm(Delta0)*V
    # display(sparse(round.(M, digits=6)))
end