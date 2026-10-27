include("../src/structs.jl")
include("../src/new_propagation.jl")
include("../src/costate_prop.jl")

using Plots 

# generators
n_qubits = 2
tmax = 2.0
tstar_min = 0
tstar_max = tmax
phase = pi
gate_type = "k1"
dt=1e-2

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
tar = build_tar_ryd(gate_type, phase, n_qubits)
#tar = build_tar(gate_type, phase, n_qubits, algebra)
stor = Storage(n_qubits, size(algebra.lie_basis, 1))
println("Setup finished")
##
dts = [1e-1, 1e-2, 1e-3]
dms = [1e-1, 1e-2, 1e-3]
d   = length(algebra.lie_basis)
m0  = rand(d)

# ── compute errors ────────────────────────────────────────────────────────────
# errs[i, j, μ] : dt = dts[i], dm = dms[j], component μ
errs = zeros(length(dts), length(dms), d)

for (i, dt) in enumerate(dts)
    tracker = costate_propagate(m0, algebra, solver, tar; dt)
    for (j, dm) in enumerate(dms), μ in 1:d
        dM_analytic = tracker.χ_t_star[μ, :]

        mp = copy(m0); mp[μ] += dm
        mm = copy(m0); mm[μ] -= dm
        tracker_mp = costate_propagate(mp, algebra, solver, tar; dt)
        tracker_mm = costate_propagate(mm, algebra, solver, tar; dt)

        errs[i, j, μ] = norm(dM_analytic - (tracker_mp.M_t_star - tracker_mm.M_t_star) / (2dm))
    end
end

# ── left column: fix dm, x-axis = dt  ────────────────────────────────────────
plots_left = map(enumerate(dms)) do (j, dm)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dt", ylabel = "‖analytic − fd‖",
             title = "dm = $dm", legend = :outerright)
    for μ in 1:d
        plot!(p, dts, errs[:, j, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dts, maximum(errs[1, j, :]) .* dts ./ dts[1];
          ls = :dash, color = :black, label = "O(dt)")
    p
end

# ── right column: fix dt, x-axis = dm  ───────────────────────────────────────
plots_right = map(enumerate(dts)) do (i, dt)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dm", ylabel = "‖analytic − fd‖",
             title = "dt = $dt", legend = :outerright)
    for μ in 1:d
        plot!(p, dms, errs[i, :, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dms, maximum(errs[i, 1, :]) .* (dms ./ dms[1]).^2;
          ls = :dash, color = :black, label = "O(dm²)")
    p
end

# ── assemble (interleave so Plots fills row-by-row into two columns) ──────────
all_plots = collect(Iterators.flatten(zip(plots_left, plots_right)))
n = length(dts)
plot(all_plots...; layout = (n, 2), size = (1400, 400n),
     margin = 5Plots.mm, plot_title = "dM/dm")