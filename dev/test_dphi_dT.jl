include("../src/structs.jl")
include("../src/new_propagation.jl")

using Plots, LaTeXStrings

# generators
n_qubits = 2
tmax = 10.0
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
println("Setup finished")

##
T0 = 6.93
dt_fixeds = [1e-2, 1e-3]
dT_fixeds = [1e-2, 1e-4]
dts = [1e-1, 5e-2, 1e-2, 5e-3, 2.5e-3]
dTs = [1e-1, 1e-2, 1e-3, 1e-4]
d   = length(algebra.lie_basis)
m0  = normalize(2 .* rand(d) .- 1)
method = Midpoint()

# ── compute errors: dφ/dT ─────────────────────────────────────────────────────
# |analytic - central fd| at a given (dt, dT); nsteps is kept fixed for T0 ± dT,
# so dt = T/nsteps scales with T and the cost is a smooth function of T
function dT_err(m0, T, dt, dT)
    nsteps = round(Int, T/dt)
    tracker = new_propagate(m0, algebra, solver, tar; T=T, nsteps=nsteps, method=method, track_min=false)
    dφ_analytic = dphi_dT(tracker)
    tracker_Tp = new_propagate(m0, algebra, solver, tar; T=T+dT, nsteps=nsteps, method=method, track_min=false)
    tracker_Tm = new_propagate(m0, algebra, solver, tar; T=T-dT, nsteps=nsteps, method=method, track_min=false)
    return abs(dφ_analytic - (tracker_Tp.min_cost - tracker_Tm.min_cost) / (2dT))
end

# errs_T_dt[k][i]: dt = dts[i],  dT = dT_fixeds[k]
# errs_T_dT[k][j]: dT = dTs[j],  dt = dt_fixeds[k]
errs_T_dt = [[dT_err(m0, T0, dt, dT_fixed) for dt in dts] for dT_fixed in dT_fixeds]
errs_T_dT = [[dT_err(m0, T0, dt_fixed, dT) for dT in dTs] for dt_fixed in dt_fixeds]

# ── observed orders ───────────────────────────────────────────────────────────
for (dT_fixed, errs_dt) in zip(dT_fixeds, errs_T_dt)
    rate_dt = [log(errs_dt[i] / errs_dt[i+1]) / log(dts[i] / dts[i+1]) for i in 1:length(dts)-1]
    println("dφ/dT  vs dt (dT = $dT_fixed): ", round.(rate_dt, digits=2))
end
for (dt_fixed, errs_dT) in zip(dt_fixeds, errs_T_dT)
    rate_dT = [log(errs_dT[j] / errs_dT[j+1]) / log(dTs[j] / dTs[j+1]) for j in 1:length(dTs)-1]
    println("dφ/dT  vs dT (dt = $dt_fixed): ", round.(rate_dT, digits=2))
end

# ── plots ─────────────────────────────────────────────────────────────────────
function plot_vs_dt(errs, label, dts, dT_fixed)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dt", ylabel = "|analytic − fd|",
             title = "$label  (dT = $dT_fixed)", legend = :outerright)
    plot!(p, dts, errs; marker = :circle, label = "T = $T0")
    plot!(p, dts, errs[1] .* (dts ./ dts[1]).^2;
          ls = :dash, color = :black, label = L"O(dt^2)")
    p
end

function plot_vs_dT(errs, label, dTs, dt_fixed)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dT", ylabel = "|analytic − fd|",
             title = "$label  (dt = $dt_fixed)", legend = :outerright)
    plot!(p, dTs, errs; marker = :circle, label = "T = $T0")
    plot!(p, dTs, errs[1] .* (dTs ./ dTs[1]).^2;
          ls = :dash, color = :black, label = L"O(dT^2)")
    p
end

p1 = plot_vs_dt(errs_T_dt[1], "dφ/dT", dts, dT_fixeds[1])
p2 = plot_vs_dt(errs_T_dt[2], "dφ/dT", dts, dT_fixeds[2])
p3 = plot_vs_dT(errs_T_dT[1], "dφ/dT", dTs, dt_fixeds[1])
p4 = plot_vs_dT(errs_T_dT[2], "dφ/dT", dTs, dt_fixeds[2])

# left column: vs dt, right column: vs dT; shared y-range within each column
ylims_dt = extrema(vcat(errs_T_dt...)) .* (0.5, 2)
ylims_dT = extrema(vcat(errs_T_dT...)) .* (0.5, 2)
plot!(p1, ylims = ylims_dt); plot!(p2, ylims = ylims_dt)
plot!(p3, ylims = ylims_dT); plot!(p4, ylims = ylims_dT)

plot(p1, p3, p2, p4; layout = (2, 2), size = (1400, 800), margin = 5Plots.mm)
