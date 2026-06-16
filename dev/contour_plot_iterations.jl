include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressMeter, JLD2, LaTeXStrings

n_qubits = 2
im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
stor = Storage(algebra.n_particles, length(algebra.lie_basis))

# params
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
tstar_threshold = 5
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_threshold)
println("Setup finished")

##
println("CMA Warm start")
dt=1e-1
m_warm_cma, m_logs_warm = find_best_initial_costate_cma_with_logging(tar, algebra, solver, stor, dt=dt, show_trace=true, sigma=1, iterations=3500)
min_dist, tstar, Ulist = propagate(m_warm_cma, algebra, solver, stor, tar, dt=dt, save_ulist=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")
##
dt=1e-2
m_main, m_logs_main = find_best_initial_costate_with_logging(tar, algebra, solver, stor, dt=dt, m0=m_warm_cma, show_trace=true, show_every=100)
min_dist, tstar, Ulist = propagate(m_main, algebra, solver, stor, tar, dt=dt, save_ulist=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")

##
##
dt = 1e-2
optimise_for_phi = pi
p_dists = plot(
            xlabel="Evolution time t",
            ylabel=L"d_{min}(t)",
            yscale=:log10,
            grid=true,
            minorgrid=true,
            gridalpha=0.5,
            minorgridalpha=0.2,
            title=latexstring("\\operatorname{Target:}\\;C_$(n_qubits-1)Phase(ϕ/π=$(optimise_for_phi/pi)), \\quad t_{optimal}>$(solver.tstar_threshold), \\quad Δt=$dt"),
            titlefontsize=12,
            legend=:bottomright
    )
gate_type = "k1"
tar = build_tar(gate_type, optimise_for_phi, n_qubits, algebra)
stor = Storage(n_qubits, length(algebra.lie_basis))
m_best, m_log = find_best_initial_costate_with_logging(tar, algebra, solver, stor, dt=dt, show_trace=true, iterations=1000)

results = []
for (i, m) in enumerate(m_log) 

    stor = Storage(n_qubits, length(algebra.lie_basis))
    d, tstar, Ulist, _ = propagate(m, algebra, solver, stor, tar, save=true, dt=dt)
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
    title=latexstring("\\operatorname{Target:}\\;C_$(n_qubits-1)Phase(ϕ/π=$(optimise_for_phi/pi),\\; $gate_type), \\quad t_{optimal}>$(solver.tstar_threshold), \\quad Δt=$dt")

    )
display(p)

##
dbest, tstarbest, Ulistbest, αlistbest = propagate(m_best, algebra, solver, stor, tar, dt=dt, save=true)
dists_best = [distance(Ulistbest[i], tar, algebra, stor) for i in eachindex(1:length(Ulistbest))]
plot!(p_dists, ts_arr[1], dists_best[1:end-1], label=latexstring("\\operatorname{Target\\; type:} $gate_type"))

display(p_dists)

##
p2 = plot(ts_arr[1], αlistbest[1:end-1])
display(p2)
##
stor.alpha = 0.0
dbest, tstarbest, Ulistbest, αlistbest = propagate(m_best, algebra, solver, stor, tar, dt=dt, save=true)
p2 = plot(ts_arr[1][1:round(Int, tstarbest/dt)], αlistbest[1:round(Int, tstarbest/dt)]/(2pi),
xlabel="Evolution time t", ylabel=L"\operatorname{Optimal\;control}\;\; \frac{\bar{α}\,(t)}{2π}", grid=true, minorgrid=true, legend=false)

# eig = abs(eigvals(Matrix(algebra.im_control))[1])
# period = 2*pi/eig
# αmodlist =  [mod(α, period) for α in αlistbest]
# plot!(p2, ts_arr[1][1:round(Int, tstarbest/dt)], αmodlist[1:round(Int, tstarbest/dt)], label="Mod solution")

title!(p2, latexstring("\\operatorname{Target:}\\;C_$(n_qubits-1)Phase(ϕ/π=$(optimise_for_phi/pi),\\; $gate_type), \\quad t_{optimal}>$(solver.tstar_threshold), \\quad Δt=$dt"), titlefontsize=12)
vline!(p2, [tstarbest], linestyle=:dash, color=:red, label=false)
annotate!(p2, 0.85 * tstarbest , 0.15, 
    text(latexstring("t_{\\operatorname{optimal}}=$(round(tstarbest/(2*pi), sigdigits=3)) \\frac{\\Omega T}{2π}"), :red, :top, 12))
display(p2)
