using Plots

proj_xyz(P, Xv, Yv, Zv) = (abs(dot(vec(P), Xv)),
                          abs(dot(vec(P), Yv)),
                          abs(dot(vec(P), Zv)))

function plot_optimal_geodesic(sol, gens, target, params, t_min)
    X = gens[2]
    Y = target
    Z = gens[1]
    target_x = abs(dot(target, X))
    target_y = abs(dot(target, Y))
    target_z = abs(dot(target, Z))

    dim = params.dim
    dim2 = params.dim2
    Xv = vec(X)
    Yv = vec(Y)
    Zv = vec(Z)

    x_vals = Vector{Float64}(undef, length(sol.u))
    y_vals = Vector{Float64}(undef, length(sol.u))
    z_vals = Vector{Float64}(undef, length(sol.u))

    for (k, u) in pairs(sol.u)
        @views Pk = reshape(view(u, 1:dim2), dim, dim)  # matrix at time k
        x_vals[k], y_vals[k], z_vals[k] = proj_xyz(Pk, Xv, Yv, Zv)
        
    end

        plt = plot3d(
        xlabel="X projection",
        ylabel="Y projection",
        zlabel="Z projection",
        title="Shortest geodesic with t = $(round(t_min / pi, digits=2)) π",
        legend=false
    )

    αgrid = range(-2*pi, 2*pi; length=100)

    x_orb = Vector{Float64}(undef, length(αgrid))
    y_orb = Vector{Float64}(undef, length(αgrid))
    z_orb = Vector{Float64}(undef, length(αgrid))

    for (j, α) in pairs(αgrid)
        U = exp((-α*im) * Matrix(Z))
        Tα = U * target * U'    
        x_orb[j], y_orb[j], z_orb[j] = proj_xyz(Tα, Xv, Yv, Zv)
    end

    plot3d!(plt, x_orb, y_orb, z_orb; linewidth=2, linestyle=:dash, color=:black)
    plot3d!(plt, x_vals, y_vals, z_vals, linewidth=2)
    scatter3d!(plt, [x_vals[1]], [y_vals[1]], [z_vals[1]], marker=:circle, label="Start")
    scatter3d!(plt, [x_vals[end]], [y_vals[end]], [z_vals[end]], marker=:diamond, label="End")
    scatter3d!(plt, [target_x], [target_y], [target_z], color=:black, markersize=6, label="Target")
    annotate!(plt, [(target_x, target_y, target_z, text("Target", :black, 12))])
    plot3d!(plt; zlims=(0, 2.0))
    savefig(plt, "1_qubit.png")

    return plt
end

# function plot_p_inner_target(sol, target, params, name)
#     dim = params.dim
#     dim2 = dim^2
#     p_vals_rot = Vector{Float64}(undef, length(sol.u))
#     p_vals_lab = Vector{Float64}(undef, length(sol.u))
#     coset_vals_rot = Vector{Float64}(undef, length(sol.u))
#     coset_vals_lab = Vector{Float64}(undef, length(sol.u))
#     tv = vec(target)

#     for k in eachindex(sol.u)
#         u = sol.u[k]
#         t = sol.t[k]

#         @views Pk = reshape(view(u, 1:dim2), dim, dim)

#         U0 = spdiagm(0 => exp.(-im .* params.V_Ryd .* t))
#         P_lab = U0 * Pk
#         p_vals_rot[k] = abs(dot(vec(Pk), tv)) / (norm(Pk) * norm(tv))
#         p_vals_lab[k] = abs(dot(vec(P_lab), tv)) / (norm(P_lab) * norm(tv))
#         coset_vals_rot[k] = 
#     end


#     plt = plot(sol.t, [p_vals_rot p_vals_lab];
#         xlabel = "t",
#         label = ["Rot frame overlap" "Lab frame overlap" "Coset dist rot" "Coset dist lab"]
#     )
    
#     #savefig(plt, "$(name)_P_target_overlap.png")

#     return plt
# end
