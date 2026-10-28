include("../src/structs.jl")
include("../src/new_propagation.jl")

using Plots, LaTeXStrings

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
dt_fixeds = [1e-2, 1e-3]
dm_fixeds = [1e-2, 1e-4]
dts = [1e-1, 5e-2, 1e-2, 5e-3, 2.5e-3]
dms = [1e-1, 1e-2, 1e-3, 1e-4]
d   = length(algebra.lie_basis)
m0  = rand(d)
method = Midpoint()

# ── compute errors: dφ/dm ─────────────────────────────────────────────────────
# |analytic - central fd| for each μ at a given (dt, dm)
function grad_errs(m0, dt, dm)
    tracker = new_propagate(m0, algebra, solver, tar; dt=dt, method=method)
    dφ_analytic = zeros(Float64, d)
    grad_infidelity!(dφ_analytic, algebra, tracker)
    errs = zeros(d)
    for μ in 1:d
        mp = copy(m0); mp[μ] += dm
        mm = copy(m0); mm[μ] -= dm
        tracker_mp = new_propagate(mp, algebra, solver, tar; dt=dt, method=method)
        tracker_mm = new_propagate(mm, algebra, solver, tar; dt=dt, method=method)
        errs[μ] = abs(dφ_analytic[μ] - (tracker_mp.min_cost - tracker_mm.min_cost) / (2dm))
    end
    return errs
end

# errs_φ_dt[k][i, μ]: dt = dts[i],  dm = dm_fixeds[k]
# errs_φ_dm[k][j, μ]: dm = dms[j],  dt = dt_fixeds[k]
errs_φ_dt = [reduce(vcat, (grad_errs(m0, dt, dm_fixed)' for dt in dts)) for dm_fixed in dm_fixeds]
errs_φ_dm = [reduce(vcat, (grad_errs(m0, dt_fixed, dm)' for dm in dms)) for dt_fixed in dt_fixeds]

# ── observed orders ───────────────────────────────────────────────────────────
for (dm_fixed, errs_dt) in zip(dm_fixeds, errs_φ_dt)
    rate_dt = [log(maximum(errs_dt[i, :]) / maximum(errs_dt[i+1, :])) /
               log(dts[i] / dts[i+1]) for i in 1:length(dts)-1]
    println("dφ/dm  vs dt (dm = $dm_fixed): ", round.(rate_dt, digits=2))
end
for (dt_fixed, errs_dm) in zip(dt_fixeds, errs_φ_dm)
    rate_dm = [log(maximum(errs_dm[j, :]) / maximum(errs_dm[j+1, :])) /
               log(dms[j] / dms[j+1]) for j in 1:length(dms)-1]
    println("dφ/dm  vs dm (dt = $dt_fixed): ", round.(rate_dm, digits=2))
end

# ── plots ─────────────────────────────────────────────────────────────────────
function plot_vs_dt(errs, label, dts, dm_fixed, d)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dt", ylabel = "|analytic − fd|",
             title = "$label  (dm = $dm_fixed)", legend = :outerright)
    for μ in 1:d
        plot!(p, dts, errs[:, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dts, maximum(errs[1, :]) .* (dts ./ dts[1]).^2;
          ls = :dash, color = :black, label = L"O(dt^2)")
    p
end

function plot_vs_dm(errs, label, dms, dt_fixed, d)
    p = plot(xscale = :log10, yscale = :log10,
             xlabel = "dm", ylabel = "|analytic − fd|",
             title = "$label  (dt = $dt_fixed)", legend = :outerright)
    for μ in 1:d
        plot!(p, dms, errs[:, μ]; marker = :circle, label = "μ = $μ")
    end
    plot!(p, dms, maximum(errs[1, :]) .* (dms ./ dms[1]).^2;
          ls = :dash, color = :black, label = L"O(dm^2)")
    p
end

p1 = plot_vs_dt(errs_φ_dt[1], "dφ/dm", dts, dm_fixeds[1], d)
p2 = plot_vs_dt(errs_φ_dt[2], "dφ/dm", dts, dm_fixeds[2], d)
p3 = plot_vs_dm(errs_φ_dm[1], "dφ/dm", dms, dt_fixeds[1], d)
p4 = plot_vs_dm(errs_φ_dm[2], "dφ/dm", dms, dt_fixeds[2], d)

# left column: vs dt, right column: vs dm; shared y-range within each column
ylims_dt = extrema(vcat(errs_φ_dt...)) .* (0.5, 2)
ylims_dm = extrema(vcat(errs_φ_dm...)) .* (0.5, 2)
plot!(p1, ylims = ylims_dt); plot!(p2, ylims = ylims_dt)
plot!(p3, ylims = ylims_dm); plot!(p4, ylims = ylims_dm)

plot(p1, p3, p2, p4; layout = (2, 2), size = (1400, 800), margin = 5Plots.mm)
