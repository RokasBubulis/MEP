using Plots

function make_plot(phase_lst, t_lst, dist_lst)
    # Separate zero and non-zero indices
    nonzero_mask = dist_lst .> 0
    zero_mask     = .!nonzero_mask

    # Build log-scaled colors only for non-zero points
    log_dist = similar(dist_lst, Float64)
    log_dist[nonzero_mask] = log10.(dist_lst[nonzero_mask])
    log_dist[zero_mask]    .= NaN   # won't affect colorbar range

    p = scatter(phase_lst[nonzero_mask] ./(π), t_lst[nonzero_mask],
        zcolor            = log_dist[nonzero_mask],
        marker            = :circle,
        markersize        = 4,
        markerstrokewidth = 0,
        colorbar_title    = "\nlog(dmin)",
        colorbar_titlefontsize = 8,
        colorbar_tickfontsize  = 7,
        right_margin      = 5Plots.mm,
        xlabel            = "ϕ/π",
        ylabel            = "t*",
        label             = false,
        c                 = :viridis,
        gridalpha         = 0.3,
        gridlinewidth     = 0.5,
        minorgrid         = true,
        minorgridalpha    = 0.15,
        minorgridlinewidth = 0.5,
    )

    # Overlay zero-distance points in black
    scatter!(p,
        phase_lst[zero_mask] ./(2π), t_lst[zero_mask],
        marker            = :circle,
        markersize        = 4,
        markerstrokewidth = 0,
        color             = :grey,
        label             = false,
    )
    return p
end 

function make_plot_multi_m(phase_lst, t_lst, dist_lst)

    nphase, npoints = size(dist_lst)

    # Repeat each phase value across columns
    phase_flat = vec(repeat(reshape(phase_lst, :, 1), 1, npoints))

    t_flat    = vec(t_lst)
    dist_flat = vec(dist_lst)

    # Separate zero and non-zero distances
    nonzero_mask = dist_flat .> 0
    zero_mask    = .!nonzero_mask

    log_dist = fill(NaN, length(dist_flat))
    log_dist[nonzero_mask] .= log10.(dist_flat[nonzero_mask])

    p = scatter(
        phase_flat[nonzero_mask] ./ π,
        t_flat[nonzero_mask],
        zcolor = log_dist[nonzero_mask],
        marker = :circle,
        markersize = 4,
        markerstrokewidth = 0,
        colorbar_title = "\nlog(dmin)",
        colorbar_titlefontsize = 8,
        colorbar_tickfontsize = 7,
        right_margin = 5Plots.mm,
        xlabel = "ϕ/π",
        ylabel = "t*",
        label = false,
        c = :viridis,
        gridalpha = 0.3,
        gridlinewidth = 0.5,
        minorgrid = true,
        minorgridalpha = 0.15,
        minorgridlinewidth = 0.5,
    )

    scatter!(
        p,
        phase_flat[zero_mask] ./ π,
        t_flat[zero_mask],
        marker = :circle,
        markersize = 4,
        markerstrokewidth = 0,
        color = :grey,
        label = false,
    )

    return p
end