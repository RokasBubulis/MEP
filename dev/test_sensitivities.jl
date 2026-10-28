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
dt_fixed = 1e-2
dm_fixed = 1e-3
dts = [1e-1, 5e-2, 1e-2, 5e-3, 2.5e-3]
dms = [1e-1, 1e-2, 1e-3, 1e-4]
d   = length(algebra.lie_basis)
m0  = rand(d)
method = Midpoint()

# ── compute errors: dU/dm ─────────────────────────────────────────────────────
errs_U_dt = zeros(length(dts), d)
errs_U_dm = zeros(length(dms), d)

for (i, dt) in enumerate(dts)
    tracker = new_propagate(m0, algebra, solver, tar; dt, method)
    for μ in 1:d
        dU_analytic = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν]
                          for ν in 1:d) * tracker.U_t_star
        mp = copy(m0); mp[μ] += dm_fixed
        mm = copy(m0); mm[μ] -= dm_fixed
        tracker_mp = new_propagate(mp, algebra, solver, tar; dt, method)
        tracker_mm = new_propagate(mm, algebra, solver, tar; dt, method)
        errs_U_dt[i, μ] = norm(dU_analytic - (tracker_mp.U_t_star - tracker_mm.U_t_star) / (2dm_fixed))
    end
end

tracker = new_propagate(m0, algebra, solver, tar; dt=dt_fixed, method)
for (j, dm) in enumerate(dms), μ in 1:d
    dU_analytic = sum(tracker.B_t_star[μ, ν] * algebra.lie_basis[ν]
                      for ν in 1:d) * tracker.U_t_star
    mp = copy(m0); mp[μ] += dm
    mm = copy(m0); mm[μ] -= dm
    tracker_mp = new_propagate(mp, algebra, solver, tar; dt=dt_fixed, method)
    tracker_mm = new_propagate(mm, algebra, solver, tar; dt=dt_fixed, method)
    errs_U_dm[j, μ] = norm(dU_analytic - (tracker_mp.U_t_star - tracker_mm.U_t_star) / (2dm))
end

# ── compute errors: dM/dm ─────────────────────────────────────────────────────
errs_M_dt = zeros(length(dts), d)
errs_M_dm = zeros(length(dms), d)

for (i, dt) in enumerate(dts)
    tracker = new_propagate(m0, algebra, solver, tar; dt, method)
    for μ in 1:d
        dM_analytic = tracker.χ_t_star[μ, :]
        mp = copy(m0); mp[μ] += dm_fixed
        mm = copy(m0); mm[μ] -= dm_fixed
        tracker_mp = new_propagate(mp, algebra, solver, tar; dt, method)
        tracker_mm = new_propagate(mm, algebra, solver, tar; dt, method)
        errs_M_dt[i, μ] = norm(dM_analytic - (tracker_mp.M_t_star - tracker_mm.M_t_star) / (2dm_fixed))
    end
end

tracker = new_propagate(m0, algebra, solver, tar; dt=dt_fixed, method)
for (j, dm) in enumerate(dms), μ in 1:d
    dM_analytic = tracker.χ_t_star[μ, :]
    mp = copy(m0); mp[μ] += dm
    mm = copy(m0); mm[μ] -= dm
    tracker_mp = new_propagate(mp, algebra, solver, tar; dt=dt_fixed, method)
    tracker_mm = new_propagate(mm, algebra, solver, tar; dt=dt_fixed, method)
    errs_M_dm[j, μ] = norm(dM_analytic - (tracker_mp.M_t_star - tracker_mm.M_t_star) / (2dm))
end

# ── observed orders ───────────────────────────────────────────────────────────
for (label, errs_dt, errs_dm) in [("dU/dm", errs_U_dt, errs_U_dm),
                                   ("dM/dm", errs_M_dt, errs_M_dm)]
    rate_dt = [log(maximum(errs_dt[i, :]) / maximum(errs_dt[i+1, :])) /
               log(dts[i] / dts[i+1]) for i in 1:length(dts)-1]
    rate_dm = [log(maximum(errs_dm[j, :]) / maximum(errs_dm[j+1, :])) /
               log(dms[j] / dms[j+1]) for j in 1:length(dms)-1]
    println("$label  vs dt: ", round.(rate_dt, digits=2),
            "  vs dm: ", round.(rate_dm, digits=2))
end

# ── plots ─────────────────────────────────────────────────────────────────────
function plot_vs_dt(errs, label, dts, dm_fixed, d)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dt", ylabel = "‖analytic − fd‖",
             title = "$label  (dm = $dm_fixed)", legend = :outerright)
    for μ in 1:d
        plot!(p, dts, errs[:, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dts, maximum(errs[1, :]) .* (dts ./ dts[1]).^2;
          ls = :dash, color = :black, label = "O(dt²)")
    p
end

function plot_vs_dm(errs, label, dms, dt_fixed, d)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dm", ylabel = "‖analytic − fd‖",
             title = "$label  (dt = $dt_fixed)", legend = :outerright)
    for μ in 1:d
        plot!(p, dms, errs[:, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dms, maximum(errs[1, :]) .* (dms ./ dms[1]).^2;
          ls = :dash, color = :black, label = "O(dm²)")
    p
end

p1 = plot_vs_dt(errs_U_dt, "dU/dm", dts, dm_fixed, d)
p2 = plot_vs_dm(errs_U_dm, "dU/dm", dms, dt_fixed, d)
p3 = plot_vs_dt(errs_M_dt, "dM/dm", dts, dm_fixed, d)
p4 = plot_vs_dm(errs_M_dm, "dM/dm", dms, dt_fixed, d)

plot(p1, p2, p3, p4; layout = (2, 2), size = (1400, 800), margin = 5Plots.mm)