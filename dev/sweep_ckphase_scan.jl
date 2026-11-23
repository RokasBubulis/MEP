# ---------------------------------------------------------------------------
# T*(Φ) and the optimal control α_opt(t) of the controlled-phase gate C-Φ, for Φ
# from π down to 0.01π, run separately for each n_qubits through run_ckphase
# (dev/ckphase.jl). This file only drives that function; nothing here touches src/.
#
#   plot  T* against Φ, colour = number of random starts needed before one reached a
#         gate with φ < success_tol ("none" = the warm start reached it, no random did).
#   store res/ckphase_scan/n<N>_T_vs_phase.{txt,png,jld2} and, per gate,
#         res/ckphase_scan/pulses/n<N>/phase_<Φ/π>pi.{jld2,png} with T*, α_opt(t), m.
#
# Why the sweep goes backwards, and why every phase also gets a "warm" start: at k >= 2
# control qubits a cold start rarely reaches the gate (measured for n = 3: 1/24 at 0.8π,
# 0/24 at 0.6π in a narrow window, 1/48 at 0.6π in a wide one), so each phase is also
# started from the previous phase's solved control — costate m and T* — which lands on
# the monotonic branch. The random starts are still drawn every phase, so the colour
# stays an honest measure of how hard the phase is.
#
# Restartable: a finished phase is appended to n<N>_T_vs_phase.txt and skipped on
# a re-run, so an interrupted sweep resumes where it stopped.
#
# Run: julia -t auto --project=. dev/sweep_ckphase_scan.jl
# ---------------------------------------------------------------------------

using Plots, LaTeXStrings, Printf, JLD2
include(joinpath(@__DIR__, "ckphase.jl"))

# ── inputs ──────────────────────────────────────────────────────────────────
n_list         = [2, 3]                 # n_qubits, swept separately
n_random       = Dict(2 => 96, 3 => 32) # random starts per phase, on top of the warm start
# Stage 1 needs a WIDE time window: J = φ + wT pulls every start down to the window's lower edge, so a
# window centred tightly on the expected T* drowns the cold search (measured for n = 3 at 0.6π:
# 0/24 gates in a ±0.4 window, 1/48 in [7, 14]). The descents are unaffected — they walk down in their
# own ±1e-6 window from wherever stage 1 lands.
t_win          = Dict(2 => (2.0, 10.0), 3 => (2.0, 13.0))
success_tol    = 1e-8                   # a start counts as a gate below this infidelity (also the colour metric)
refine_tol     = 1e-8                   # threshold the descent has to hold; same as res/HJB_T_vs_phase.txt
run_time_limit = 180.0                  # abandon a single run after this many seconds
seed           = 1
max_attempts   = 6                     # redraws allowed per phase before it is recorded as "no monotone gate"
t_gap          = 1e-3                 # a phase must beat the previous T* by at least this to be accepted

# jld2 holding the Φ = π gate whose costate m and T* seed the chain; T*(π) = 7.6114 (n=2), 10.9636 (n=3)
seed_file = Dict(2 => "ckphase_k1_phase_1pi_control.jld2", 3 => "ckphase_k2_phase_1pi_control.jld2")

# 0.05π down to 0.10π, then finer where T* turns over
phases = Float64[collect(range(float(π), 0.10π, step = -0.05π)); 0.08π; 0.05π; 0.03π; 0.02π; 0.01π]

out_root   = joinpath(@__DIR__, "..", "res", "ckphase_scan")
pulse_root = joinpath(out_root, "pulses")
mkpath(out_root)

# a gate below refine_tol wins; among gates the shorter pulse wins, otherwise the lower φ
function better(a, b)
    ga, gb = a.φ < refine_tol, b.φ < refine_tol
    ga != gb && return ga ? a : b
    return ga ? (a.T <= b.T ? a : b) : (a.φ <= b.φ ? a : b)
end

for nq in n_list
    k       = nq - 1
    nrand   = n_random[nq]
    summary = joinpath(out_root, @sprintf("n%d_T_vs_phase.txt", nq))
    pulses  = joinpath(pulse_root, @sprintf("n%d", nq))
    mkpath(pulses)

    # phases already solved by an earlier (possibly interrupted) run, and their solutions, so that a
    # resumed sweep still has a warm start for the next phase
    done  = Dict{Float64,Any}()
    if isfile(summary)
        for ln in eachline(summary)
            (startswith(ln, "#") || isempty(strip(ln))) && continue
            v = split(ln)
            f = parse(Float64, v[1])
            pf = joinpath(pulses, @sprintf("phase_%gpi.jld2", f))
            if v[4] == "true" && isfile(pf)
                d = load(pf)
                done[f] = (m = d["m"], T = d["T"])
            else
                done[f] = nothing
            end
        end
    end
    isempty(done) && open(summary, "w") do io
        println(io, "# C-Φ, n_qubits = $nq (k = $k control qubits): Φ/π  T*Ω  infidelity  gate  " *
                    "random starts needed (of $nrand; $nrand+1 = none, the warm start delivered the gate)  " *
                    "random starts that reached the gate")
    end

    # the chain starts at Φ = π, whose gate is already known
    sd = load(joinpath(@__DIR__, "..", "res", seed_file[nq]))
    prev = (m = sd["m"], T = Float64(sd["T"]))

    for Φ in phases
        key = round(Φ/π, digits = 4)
        if haskey(done, key)
            done[key] === nothing || (prev = done[key])
            continue
        end

        # T* must fall with Φ. A gate is accepted only if it also beats the previous phase's T* by more than
        # t_gap; otherwise the phase is redrawn (fresh seed, same window — the window is never shrunk, so a
        # hard phase can still be found). A phase that never satisfies both is recorded with gate = false.
        # The first phase has no predecessor to beat: Φ = π is seeded with its own solved gate, so its T*
        # cannot be lower than prev.T and the monotone test would reject it forever.
        first_phase = Φ == first(phases)
        gate = false; shots_total = 0; n_gates = 0; r = nothing; attempt = 0
        while !gate && attempt < max_attempts
            attempt += 1
            r = run_ckphase(; phase = Φ, n_qubits = nq, n_random = nrand, max_runs = nrand + 1,
                              target_hits = nrand + 1, success_tol, refine_tol,
                              tstar_min = t_win[nq][1], tstar_max = t_win[nq][2],
                              warm_starts = ((m = prev.m, T = prev.T),),
                              run_time_limit, seed = seed + attempt - 1, save_results = false)
            # colour = share of RANDOM starts that reached the gate, warm start excluded, summed over attempts.
            # shots = how many random starts came before the first one that reached the gate (kept for the txt)
            first_gate   = findfirst(p -> !p.warm && p.φ < success_tol, r.polished)
            shots_total += first_gate === nothing ? nrand : first_gate - 1
            n_gates     += count(p -> !p.warm && p.φ < success_tol, r.polished)
            gate = r.φ < refine_tol && (first_phase || r.T < prev.T - t_gap)
            gate || @printf("[n=%d] Φ = %.4gπ  attempt %d: T* = %.4f does not beat %.4f  (φ = %.2e); redrawing\n",
                            nq, key, attempt, r.T, prev.T, r.φ)
        end
        shots = min(shots_total, nrand + 1)


        open(summary, "a") do io
            @printf(io, "%.4f  %.4f  %.2e  %-5s  %d  %d\n", key, r.T, r.φ, gate, shots, n_gates)
        end
        @printf("[n=%d] Φ = %.4gπ  T* = %.4f  φ = %.2e  %s  %s  (%d random starts reached the gate in %d attempt(s))\n",
                nq, key, r.T, r.φ, gate ? "gate   " : "NO GATE",
                shots > nrand ? "none did; the warm start did" : "$shots random start(s) needed",
                n_gates, attempt)
        flush(stdout)

        if gate
            tag = @sprintf("phase_%gpi", key)
            jldsave(joinpath(pulses, tag * ".jld2"); phase = Φ, k, T = r.T, infidelity = r.φ,
                    t = r.t, alpha = r.α, m = r.m)
            savefig(r.plot, joinpath(pulses, tag * ".png"))
            prev = (m = r.m, T = r.T)
        end
    end

    # ── plot: T* against Φ, colour = % of random starts that reached the gate ────
    # a phase interrupted mid-append can leave a duplicate line; the last one wins
    rows = Dict{Float64,NamedTuple}()
    for ln in eachline(summary)
        (startswith(ln, "#") || isempty(strip(ln))) && continue
        v = split(ln)
        rows[parse(Float64, v[1])] = (Φπ = parse(Float64, v[1]), T = parse(Float64, v[2]),
                                      φ = parse(Float64, v[3]), gate = v[4] == "true",
                                      shots = parse(Int, v[5]), gates = parse(Int, v[6]))
    end
    rows = sort!(collect(values(rows)), by = r -> -r.Φπ)
    Φlist = [r.Φπ for r in rows]; Tlist = [r.T for r in rows]
    g  = [r.gate for r in rows]; s = [r.shots for r in rows]
    frac = [100*r.gates/nrand for r in rows]    # % of random starts that reached the gate
    n_gate = count(g)

    # y range from the data: :auto freezes the limits at plot() construction, before any series exists
    lo, hi = isempty(Tlist) ? (0.0, 1.0) : extrema(Tlist)
    pad = 0.06*max(hi - lo, 1e-3)
    fmax = isempty(frac) ? 1.0 : max(1.0, maximum(frac))
    ylo, yhi = lo - pad, hi + pad

    # legend corner: pick the corner with no markers in it rather than hardcoding one — the curve runs
    # bottom-left -> top-right, so a top-left legend clears n=2 but n=3's mid-curve rises through that box.
    # Each candidate box is ~1/3 of the axes wide and ~1/4 tall.
    fx = (Φlist .- (-0.02)) ./ 1.05
    fy = (Tlist .- ylo) ./ (yhi - ylo)
    cands = ((:topleft,  (fx .< 0.34) .& (fy .> 0.72)), (:topright, (fx .> 0.66) .& (fy .> 0.72)),
             (:bottomleft, (fx .< 0.34) .& (fy .< 0.28)), (:bottomright, (fx .> 0.66) .& (fy .< 0.28)))
    nocc = [sum(v) for (_, v) in cands]
    legcorner = cands[argmin(nocc)][1]
    nocc = minimum(nocc)
    nocc > 0 && @printf("[n=%d] warning: %d marker(s) under the %s legend\n", nq, nocc, legcorner)
    p = plot(xlims = (-0.02, 1.03), ylims = (ylo, yhi),
             xlabel = L"\Phi/\pi", ylabel = L"T^*\Omega",
             title = @sprintf("C-Φ, %d qubits: minimal pulse time vs phase", nq), titlefontsize = 10,
             legend = legcorner,
             framestyle = :box, mirror = false,       # all four spines, tick labels left/bottom only
             grid = true, minorgrid = true, minorticks = 5,
             gridalpha = 0.35, gridlinewidth = 0.6, minorgridalpha = 0.15, minorgridlinewidth = 0.4,
             size = (700, 460), right_margin = 8Plots.mm, left_margin = 5Plots.mm, bottom_margin = 4Plots.mm)
    # colour = fraction of random starts that reached the gate; "starts needed" saturates (n=2: 1..9 on 0..96)
    any(g) && scatter!(p, Φlist[g], Tlist[g]; marker_z = frac[g], clims = (0.0, fmax), color = :viridis,
                       markersize = 4, markerstrokewidth = 0, label = "gate, φ < $(refine_tol)",
                       colorbar_title = "random starts reaching\nthe gate (%)")
    any(.!g) && scatter!(p, Φlist[.!g], Tlist[.!g]; marker = :xcross, color = :gray, markersize = 4,
                         markerstrokewidth = 0, label = "no gate (best T)")

    png = joinpath(out_root, @sprintf("n%d_T_vs_phase.png", nq))
    savefig(p, png)
    jldsave(joinpath(out_root, @sprintf("n%d_T_vs_phase.jld2", nq)); n_qubits = nq, k, n_random = nrand,
            Φπ = Φlist, T = Tlist, φ = [r.φ for r in rows], gate = g, shots = s,
            gates = [r.gates for r in rows], success_tol, refine_tol)
    @printf("[n=%d] %d/%d phases reached φ < %.0e; saved %s and the pulses in %s\n",
            nq, n_gate, length(rows), refine_tol, png, pulses)
end

println("done: summaries and plots in $out_root, one jld2 + png per gate in $pulse_root/n<N>/")
