include("../src/structs.jl")
include("../src/new_propagation.jl")

using Plots

# generators
n_qubits = 2
tmax = 2.0
tstar_min = 0
tstar_max = tmax
phase = pi
gate_type = "k1"

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
tar = build_tar_ryd(gate_type, phase, n_qubits)
stor = Storage(n_qubits, size(algebra.lie_basis, 1))
println("Setup finished")
##

# exact halvings so the observed order is unambiguous
dts = [1e-1, 5e-2, 1e-2, 5e-3, 2.5e-3]
# small dm: central-difference truncation is O(dm^2) and must stay below the dt error
dms = [1e-4, 1e-5]
d   = length(algebra.lie_basis)
m0  = rand(d)

# ── compute errors ────────────────────────────────────────────────────────────
# errs[i, j, μ] : dt = dts[i], dm = dms[j], component μ
errs = zeros(length(dts), length(dms), d)

for (i, dt) in enumerate(dts)
    tracker = new_propagate(m0, algebra, solver, tar; dt)
    for (j, dm) in enumerate(dms), μ in 1:d
        dU_analytic = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν]
                          for ν in 1:d) * tracker.U_t_star

        mp = copy(m0); mp[μ] += dm
        mm = copy(m0); mm[μ] -= dm
        tracker_mp = new_propagate(mp, algebra, solver, tar; dt)
        tracker_mm = new_propagate(mm, algebra, solver, tar; dt)

        errs[i, j, μ] = norm(dU_analytic - (tracker_mp.U_t_star - tracker_mm.U_t_star) / (2dm))
    end
end

# observed order across each bracket: log(err[i]/err[i+1]) / log(dt[i]/dt[i+1]);
# ≈ 2 for every bracket confirms global O(dt²) state propagation
for (j, dm) in enumerate(dms)
    rate = [log(maximum(errs[i, j, :]) / maximum(errs[i+1, j, :])) /
            log(dts[i] / dts[i+1]) for i in 1:length(dts)-1]
    println("dm = $dm   observed order = ", round.(rate, digits = 2))
end

# ── left column: fix dm, x-axis = dt ─────────────────────────────────────────
plots_left = map(enumerate(dms)) do (j, dm)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dt", ylabel = "‖analytic − fd‖",
             title = "dm = $dm", legend = :outerright)
    for μ in 1:d
        plot!(p, dts, errs[:, j, μ]; marker = :circle, label = "μ = $μ")
    end
    # reference line with slope 2 (second order), anchored at the largest-dt max error
    plot!(p, dts, maximum(errs[1, j, :]) .* (dts ./ dts[1]).^2;
          ls = :dash, color = :black, label = "O(dt²)")
    p
end

# ── right column: fix dt, x-axis = dm ────────────────────────────────────────
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

# ── assemble ──────────────────────────────────────────────────────────────────
all_plots = collect(Iterators.flatten(zip(plots_left, plots_right)))
plot(all_plots...; layout = (length(dms), 2), size = (1200, 300 * length(dms)),
     margin = 5Plots.mm, plot_title = "dU/dm")