# Blind (cold) search for the Φ = π anchor of the n = 3 sweep.
#
# The sweep seeded Φ = π from an earlier run done in a [5, 18] window; this script re-derives that point with
# NO warm start and NO seeded costate, so the anchor of the chain is a genuine cold result. Every attempt is one
# batch of n_random random starts (random m, random T uniform in the window) followed by the engine's own
# polish (stage 1c, w = 0) and T descent (stage 2). Attempts repeat with a fresh RNG seed until a cold start
# lands on the expected branch, T* ≈ 10.96.
#
# The expected value is used ONLY to recognise the right branch; it never seeds a start.
#
# Run: julia -t 16 --project=. dev/cold_pi_n3.jl

include(joinpath(@__DIR__, "ckphase.jl"))

nq          = 3
k           = nq - 1
Φ           = float(π)
nrand       = 32
twin        = (2.0, 13.0)
success_tol = 1e-8
refine_tol  = 1e-8
expected    = 10.9636          # expected T*, for branch recognition only
branch_tol  = 0.05
max_attempts = 60

out_root = joinpath(@__DIR__, "..", "res", "ckphase_scan")
summary  = joinpath(out_root, "n3_T_vs_phase.txt")
pulses   = joinpath(out_root, "pulses", "n3")

function main()
    attempts = NamedTuple[]
    best = nothing
    println("cold search: n = $nq, Φ = π, window $twin, $nrand random starts per attempt, no warm start")
    flush(stdout)

    for a in 1:max_attempts
        t0 = time()
        r = run_ckphase(; phase = Φ, n_qubits = nq, n_random = nrand, max_runs = nrand,
                          target_hits = 1, success_tol, refine_tol,
                          tstar_min = twin[1], tstar_max = twin[2],
                          run_time_limit = 180.0, seed = 1000 + a, save_results = false)
        wall = time() - t0
        ngate = count(p -> p.φ < success_tol, r.polished)
        first_gate = findfirst(p -> p.φ < success_tol, r.polished)
        push!(attempts, (a = a, n_gate = ngate, T = r.T, φ = r.φ, wall = wall, ok = r.φ < refine_tol))
        @printf("[attempt %2d] %2d/%d starts gated   T* = %.4f  φ = %.2e   %.0f s\n",
                a, ngate, r.n_runs, r.T, r.φ, wall)
        flush(stdout)

        (r.φ < refine_tol && abs(r.T - expected) < branch_tol) || continue
        best = (r = r, a = a, first_gate = first_gate, ngate = ngate)
        break
    end

    best === nothing && error("no cold attempt reached T* ≈ $expected within $max_attempts attempts")

    r = best.r
    @printf("\ncold anchor found on attempt %d: T* = %.4f  φ = %.2e  (expected %.4f, Δ = %.1e)\n",
            best.a, r.T, r.φ, expected, r.T - expected)
    @printf("%d of %d random starts gated; the first gate was start #%d\n", best.ngate, r.n_runs, best.first_gate)

    g = r.polished[best.first_gate]
    @printf("winning start: T_start = %.4f, raw φ = %.2e → polished φ = %.2e (w = 0), status %s, promoted %s\n",
            g.T, g.φ_raw, g.φ, g.status, g.promoted)
    @printf("descent lowered T from that gate to T* = %.4f at refine_tol = %.0e\n", r.T, refine_tol)
    @printf("cold wall time: %.0f s over %d attempt(s)\n", sum(x.wall for x in attempts), length(attempts))

    # ── overwrite the Φ = π row, keeping the 6-column format, and append provenance ──
    marker = "# ── provenance of the Φ = π point (cold search, no warm start)"
    lines  = readlines(summary)
    imi    = findfirst(l -> startswith(l, marker), lines)
    imi !== nothing && (lines = lines[1:imi-1])

    rowidx = findfirst(l -> begin
        startswith(l, "#") && return false
        s = split(l)
        !isempty(s) && length(s) >= 2 && parse(Float64, s[1]) == 1.0
    end, lines)
    rowidx === nothing && error("Φ = π row not found in $summary")

    shots = best.first_gate
    lines[rowidx] = @sprintf("%.4f  %.4f  %.2e  %-5s  %d  %d", 1.0, r.T, r.φ, "true", shots, best.ngate)

    push!(lines, marker)
    push!(lines, "# obtained blind: no warm start, no seeded costate; random m, T ~ U($(twin[1]+0.3), $(twin[2])-0.3)")
    push!(lines, @sprintf("# %d batch(es) of %d random starts (fresh RNG seed each); accepted on attempt %d: %d start(s) gated",
                          length(attempts), nrand, best.a, best.ngate))
    push!(lines, @sprintf("# winning random start: T_start = %.4f, φ_raw = %.2e", g.T, g.φ_raw))
    push!(lines, @sprintf("# polishing: stage 1c, BFGS at w = 0 → φ = %.2e (status %s)", g.φ, g.status))
    push!(lines, @sprintf("# descent: stage 2, T lowered in 0.1 steps + 11 bisections → T* = %.4f, φ = %.2e",
                          r.T, r.φ))
    push!(lines, @sprintf("# cold search wall %.0f s", sum(x.wall for x in attempts)))
    write(summary, join(lines, "\n") * "\n")

    # ── overwrite the stored pulse so the anchor jld2 matches the row ──
    tag = "phase_1pi"
    jldsave(joinpath(pulses, tag * ".jld2"); phase = Φ, k, T = r.T, infidelity = r.φ,
            t = r.t, alpha = r.α, m = r.m, cold = true, n_random = nrand, n_runs = r.n_runs,
            attempts = length(attempts), first_shot = shots, n_gate = best.ngate,
            T_start = g.T, φ_raw = g.φ_raw, status = g.status)
    savefig(r.plot, joinpath(pulses, tag * ".png"))
    println("\noverwrote Φ = π in $summary and $pulses/$tag.{jld2,png}")
end

main()
