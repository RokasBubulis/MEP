
function generate_save_plots(gate_type, phase, n_qubits, algebra, title_str, file_name)
    tar = build_tar(gate_type, phase, n_qubits, algebra)
    m_warm, dist, m_log = find_best_initial_costate_bbo(tar, algebra, solver, dt=dt, verbose=verbose, max_evals=4000, logging=true)
    m_best = find_best_initial_costate(tar, algebra, solver, dt=dt, m0=m_warm, show_trace=verbose)
    min_dist, tstarbest, Ulistbest, αlistbest = propagate(m_best, algebra, solver, tar, dt=dt, save=true, show_trace=verbose)
    println("dt: $dt, dmin=$min_dist, t*=$tstarbest")

    # dists plot 
    stor = Storage(algebra.n_particles, length(algebra.lie_basis))
    ts = dt .*(1:length(Ulistbest))
    dists_best = [distance(Ulistbest[i], tar, algebra, stor) for i in eachindex(1:length(Ulistbest))]
    p1 = plot(ts, dists_best[1:end], yscale=:log10)
    title!(p1, title_str, title_font_size=12)
    savefig(p1, file_name *"dist_vs_time.png")
    display(p1)

    # control plot
    p2 = plot(ts[1:round(Int, tstarbest/dt)], αlistbest[1:round(Int, tstarbest/dt)]/(2pi),
    xlabel="Evolution time t", ylabel=L"\operatorname{Optimal\;control}\;\; \frac{\bar{α}\,(t)}{2π}", grid=true, minorgrid=true, legend=false)
    title!(p2, title_str, titlefontsize=12)
    vline!(p2, [tstarbest], linestyle=:dash, color=:red, label=false)
    annotate!(p2, 0.85 * tstarbest , (maximum(αlistbest/(2pi)) + minimum(αlistbest/(2pi)))/2, 
        text(latexstring("t_{\\operatorname{optimal}}=$(round(tstarbest/(2*pi), sigdigits=3)) \\frac{\\Omega T}{2π}"), :red, :top, 12))
    savefig(p2, file_name *"control_vs_time.png")
    display(p2)

    # iterations plot
    results = []
    for (i, m) in enumerate(m_log) 

        stor = Storage(n_qubits, length(algebra.lie_basis))
        d, tstar, Ulist, _ = propagate(m, algebra, solver, tar, save=true, dt=dt)
        n = length(Ulist)
        ts = dt .* (1:n)
        dists = [distance(Ulist[i], tar, algebra, stor) for i in eachindex(ts)]
        push!(results, (i=i, m=m, d=d, tstar=tstar, Ulist=Ulist, dists=dists, ts=ts))
    end 

    t_min_idx = 1
    t_max_idx = round(Int(tmax/dt))
    d_arr      = [r.d       for r in results]
    ts_arr     = [r.ts[t_min_idx:t_max_idx]      for r in results]
    dists_arr  = [r.dists[t_min_idx:t_max_idx]   for r in results]
    i_arr = [r.i for r in results]

    D = hcat([log10.(v) for v in dists_arr]...)
    p = contourf(i_arr, ts_arr[1], D;
        colorbar=true,
        c=:viridis,
        xlabel="Objective calls", 
        ylabel="Evolution time t",
        colorbar_title=L"\log{d_{min}(t)}",
        size=(800, 500),
        levels=50,
        title=title_str
        )
    savefig(p, file_name *"opt_iters_log.png")
    display(p)

    Dnonlong = hcat([v for v in dists_arr]...)
    p3 = contourf(i_arr, ts_arr[1], Dnonlong;
        colorbar=true,
        c=:viridis,
        xlabel="Objective calls", 
        ylabel="Evolution time t",
        colorbar_title=L"d_{min}(t)",
        size=(800, 500),
        levels=50,
        title=title_str
        )
    savefig(p3, file_name *"opt_iters.png")
    display(p3)
end 
