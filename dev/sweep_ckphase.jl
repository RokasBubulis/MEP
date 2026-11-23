# T* and optimal control α_opt(t) of the controlled-phase gate C-Φ for Φ from π down to 0.01π,
# running run_ckphase (dev/ckphase.jl) once per phase with n_random random starts.
#   plot 1: T* against Φ, colour = fraction of random starts that reached infidelity < success_tol
#   plot 2: α_opt(t) of the gates found, for a selection of phases
# Output in res/, prefixed with system_name (e.g. res/HJB_T_vs_phase_success.png).
# Run:  julia -t auto --project=. dev/sweep_ckphase.jl      (Ctrl+C stops it)
# For another system set generator_fn and system_name before including this file (see dev/sweep_ckphase_PXP.jl).

using Plots, LaTeXStrings, Printf, JLD2
include(joinpath(@__DIR__, "ckphase.jl"))

@isdefined(system_name) || (system_name = "HJB")
@isdefined(generator_fn) || (generator_fn = construct_HJB_generators)
Φ_sweep  = [collect(range(π, 0.05π, step=-0.05π)); 0.025π; 0.01π]
Φ_alpha  = [π, 0.8π, 0.6π, 0.4π, 0.2π, 0.05π, 0.01π]    # phases shown in the α plot (if a gate was found)
n_random = 96

results = []
t_sweep = time()
for Φ in Φ_sweep
    r = run_ckphase(; phase = Float64(Φ), n_random, generator_fn, save_results = false)
    # gate found = at least one start reached φ < success_tol (T* is then obtained by descent, where φ ends ≈ refine_tol,
    # possibly a hair above it, so φ < success_tol would wrongly reject some gates)
    push!(results, (Φ = Φ, T = r.T, φ = r.φ, gate = r.n_success > 0, n_success = r.n_success,
                    n_runs = r.n_runs, success = r.success, t = r.t, α = r.α, tol = r.success_tol,
                    φ_starts = [p.φ for p in r.polished]))
    @printf("[%s sweep] Φ = %.3fπ  T* = %.4f  φ = %.1e  success %d/%d = %.1f%%  (elapsed %d min)\n",
            system_name, Φ/π, r.T, r.φ, r.n_success, r.n_runs, 100r.success, round(Int, (time() - t_sweep)/60))
    flush(stdout)
end

success_tol = results[1].tol                       # the engine's threshold, for the plot label and the txt header

# ── plot 1: T* vs Φ coloured by success ratio ─────────────────────────────────
Φr  = [r.Φ for r in results]
Tr  = [r.T for r in results]
suc = [100r.success for r in results]
g   = [r.gate for r in results]
# scatter with marker_z on an empty selection crashes the GR legend, so the "gate found" series is only
# added when at least one phase produced a gate
p1 = plot(xlims = (0, 1.03), ylims = (0, 10), xlabel = L"\Phi/\pi", ylabel = L"T^*\Omega",
          title = "$(system_name): minimal gate time, $(n_random) random starts per batch", titlefontsize = 10,
          legend = :bottomright, size = (650, 420), right_margin = 5Plots.mm)
any(g) && scatter!(p1, Φr[g] ./ π, Tr[g]; marker_z = suc[g], clims = (0, 100), color = :viridis, markersize = 6, markerstrokewidth = 0,
                   label = "gate found", colorbar_title = "successful random starts (%), φ < $(success_tol)")
any(.!g) && scatter!(p1, Φr[.!g] ./ π, Tr[.!g]; marker = :xcross, color = :gray, markersize = 6, label = "no gate (best T)")

# ── plot 2: α_opt(t) for selected phases ──────────────────────────────────────
p2 = plot(xlabel = L"t\,\Omega", ylabel = L"\alpha_{\mathrm{opt}}(t) - \alpha_{\mathrm{opt}}(0)",
          title = "$(system_name): optimal controls", titlefontsize = 10, legend = :outerright, size = (750, 420))
for Φ in Φ_alpha
    k = findfirst(r -> abs(r.Φ - Φ) < 1e-9 && r.gate, results)
    k === nothing && continue
    r = results[k]
    plot!(p2, r.t, r.α .- r.α[1]; lw = 2, label = @sprintf("Φ = %.3gπ, T* = %.3f", r.Φ/π, r.T))
end

# ── save ──────────────────────────────────────────────────────────────────────
out = joinpath(@__DIR__, "..", "res", system_name)
savefig(p1, out * "_T_vs_phase_success.png")
savefig(p2, out * "_alpha_controls.png")
# φ_starts is ragged now (a phase stops once target_hits is met), so pad with NaN to a common width
φw = maximum(r.n_runs for r in results)
φ_starts = fill(NaN, φw, length(results))
for (j, r) in enumerate(results)
    φ_starts[1:r.n_runs, j] .= r.φ_starts
end
jldsave(out * "_T_vs_phase.jld2"; system_name, Φ = Φr, T = Tr, φ = [r.φ for r in results], gate = g,
        n_success = [r.n_success for r in results], n_runs = [r.n_runs for r in results],
        n_random, success_tol,
        t = [r.t for r in results], alpha = [r.α for r in results], φ_starts)
open(out * "_T_vs_phase.txt", "w") do io
    println(io, "# $(system_name): Φ/π   T*Ω   infidelity   gate   successful starts (of n_runs, φ < $(success_tol))")
    for r in results
        @printf(io, "%.4f  %.4f  %.2e  %-5s  %d/%d\n", r.Φ/π, r.T, r.φ, r.gate, r.n_success, r.n_runs)
    end
end
println("[$(system_name) sweep] done: saved $(out)_T_vs_phase_success.png, _alpha_controls.png, _T_vs_phase.jld2, _T_vs_phase.txt")
