# Minimal gate time T* and optimal control α_opt(t) of the controlled-phase gate C-Φ for ONE phase Φ.
# Library behind dev/run_ckphase.jl (one phase) and dev/sweep_ckphase.jl (a sweep of phases).
#
#   1a. n_random random starts in parallel, BFGS on J = φ + w T; no polish yet
#   1b. the n_polish_best lowest-φ runs are promoted to polishing regardless of their infidelity,
#       and any run that already landed within loose_tol of a gate (the old rule)
#   1c. the promoted runs are polished with w = 0
#   2.  from the n_descend shortest distinct gates, T is lowered at fixed T (step + bisection) against
#       the stricter refine_tol, so that the reported T* is resolved to ~T_step/2^n_bisect
#   3.  the shortest pulse meeting refine_tol is reported; its α_opt(t) is plotted
#
# Every BFGS call goes through run_bfgs_guarded: a run is aborted once J has not improved by more than
# stall_rel·|J| for stall_iters iterations, and is reported with status "stall" (vs "ok" for a converged
# run, "time" for one that hit run_time_limit, "iters" for one that hit its iteration cap). Nothing is
# thrown away: an aborted run keeps the
# best point it reached.
#
# save_results gates all three artifacts: res/c_<k>_phase_<Φ/π>π_report.txt (text report),
# _control.jld2 (T, m, α(t), …) and _control.png (the control law).
#
# Run: julia -t auto --project=. dev/run_ckphase.jl

if !@isdefined(Algebra)
    include(joinpath(@__DIR__, "..", "src", "structs.jl"))
    include(joinpath(@__DIR__, "..", "src", "propagation.jl"))
    include(joinpath(@__DIR__, "..", "src", "optimise.jl"))
end

using Plots, LaTeXStrings, Random, Printf, JLD2, LinearAlgebra

# src/optimise.jl's run_bfgs, plus a callback that stops BFGS once J has not improved by more than
# stall_rel·|J| for stall_iters iterations. Returns (result, status) with status
# "stall" (callback), "ok" (converged) or "iters" (iteration cap hit).
function run_bfgs_guarded(algebra::Algebra, solver::SolverParams, ϕ::Float64, m0, T0;
                          time_weight = 0.0, nsteps = 1000, iterations = 300,
                          g_tol = 1e-6, f_reltol = 1e-8, stall_iters = 40, stall_rel = 1e-4,
                          time_limit = 120.0)
    cg! = make_optimisation_problem(algebra, solver, ϕ; time_weight = time_weight, nsteps = nsteps)
    x0  = [normalize(m0); τ_of_T(T0, solver.tstar_min, solver.tstar_max)]
    # the callback receives the solver's state (BFGSState: .f_x), not an OptimizationState; it is called once
    # before the loop and after every accepted iteration, so the iteration index is counted here
    J_best, it_best, it, stalled = Ref(Inf), Ref(0), Ref(0), Ref(false)
    callback = state -> begin
        it[] += 1
        J = state.f_x
        if J < J_best[] - stall_rel*abs(J_best[])
            J_best[], it_best[] = J, it[]
        elseif it[] - it_best[] > stall_iters
            stalled[] = true
            return true
        end
        return false
    end
    opts = Optim.Options(; iterations = iterations, g_tol = g_tol, f_reltol = f_reltol,
                           callback = callback, time_limit = time_limit)
    res = Optim.optimize(only_fg!(cg!), x0, BFGS(), opts)
    tc = Optim.termination_code(res)
    status = stalled[] ? "stall" :
             tc == Optim.TerminationCode.Time ? "time" :
             Optim.converged(res) ? "ok" : "iters"
    return res, status
end

function run_ckphase(; phase = float(π), n_random = 48, n_qubits = 4,
                       tstar_min = 10.0, tstar_max = 15.0,
                       generator_fn = construct_HJB_generators, save_results = false,
                       success_tol = 1e-3,      # a run counts as a hit below this infidelity
                       target_hits = 1,         # keep launching batches until this many hits …
                       max_runs = n_random,     # … or this many runs in total
                       run_time_limit = 120.0,  # abandon a single run after this many seconds
                       stall_iters = 40,        # abandon a single run after this many iterations without progress
                       warm_starts = (),        # extra starts (named tuples m, T) tried before the random ones
                       # ── deeper settings ──
                       stall_rel = 1e-4, n_polish_best = 4,
                       dt = 1e-2, seed = 1, loose_frac = 0.1,
                       refine_tol = 1e-8, n_descend = 4, distinct_T = 0.02, time_weight = 1e-3,
                       T_step = 0.1, n_bisect = 11, tstar_floor = 0.5,
                       reltol = 1e-8, abstol = 1e-8, dist_tol = 1e-8, grad_tol = 1e-8, unitary_tol = 1e-8,
                       iters_start = 300, iters_fixed = 300, polish_iters = 1000, g_tol = 1e-6,
                       plot_size = (600, 380))

    k = n_qubits - 1
    im_control, im_drift = im .* generator_fn(n_qubits)
    algebra = Algebra(im_control, im_drift)
    d = length(algebra.lie_basis)

    # infidelity of doing nothing (U = I) for C-Φ: 1 - cos(Φ/4). The time weight and the first-stage
    # acceptance threshold are taken relative to it, so that at small Φ the identity neither beats a gate on J
    # nor passes as a candidate
    φ_id      = 1 - cos(phase/4)
    w         = time_weight * φ_id/(1 - cos(π/4))
    loose_tol = min(1e-2, loose_frac*φ_id)

    res_dir = joinpath(@__DIR__, "..", "res")
    tag = @sprintf("c_%d_phase_%gpi", k, round(phase/π, digits=4))
    report_file = joinpath(res_dir, tag * "_report.txt")
    plot_file   = joinpath(res_dir, tag * "_control.png")

    println("Setup finished: k = $k, Φ = $(round(phase/π, digits=4))π, $(Threads.nthreads()) threads, d = $d")

    # ── helpers ───────────────────────────────────────────────────────────────
    # optimise m at fixed T (window T ± 1e-6, so the time weight is irrelevant); returns (m, infidelity, status)
    function fixed_T(m, T, iters = iters_fixed)
        s = SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6)
        r, status = run_bfgs_guarded(algebra, s, phase, m, T; time_weight = 0.0,
                                     nsteps = ceil(Int, T/dt), iterations = iters, g_tol = g_tol,
                                     stall_iters = stall_iters, stall_rel = stall_rel,
                                     time_limit = run_time_limit)
        return minimizer_mT(r, s)[1], Optim.minimum(r), status
    end

    # infidelity of (m, T) on the T/dt grid used by fixed_T
    cost_at(m, T) = propagate(m, algebra, SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6),
                              phase; T=T, nsteps=ceil(Int, T/dt)).min_cost

    # lower T from a gate (m, T) while the gate still meets refine_tol, then bisect between the last gate and
    # the first failure. A step that misses refine_tol is retried with a longer polish before it counts as a
    # failure — unless the first attempt stalled, in which case it has converged to a local minimum above the
    # threshold and replaying it from the same start stalls in the same place.
    function descend(m, T)
        T_ok, T_bad, status = T, NaN, "ok"
        step_failed(T_try) = begin
            m1, φ1, st1 = fixed_T(m, T_try)
            (φ1 < refine_tol || st1 == "stall") && return (φ1 < refine_tol, m1, st1)
            m2, φ2, st2 = fixed_T(m, T_try, polish_iters)
            return (φ2 < refine_tol, m2, st2)
        end
        while T_ok - T_step > tstar_floor
            ok, m_k, status = step_failed(T_ok - T_step)
            ok || (T_bad = T_ok - T_step; break)
            m, T_ok = m_k, T_ok - T_step
        end
        isnan(T_bad) && return m, T_ok, status
        for _ in 1:n_bisect
            T_mid = (T_ok + T_bad)/2
            ok, m_k, status = step_failed(T_mid)
            ok ? ((m, T_ok) = (m_k, T_mid)) : (T_bad = T_mid)
        end
        return m, T_ok, status
    end

    # evolution with the found control, from the Lie costate: (tracker, t, α)
    function control_law(m, T)
        α_hist = Tuple{Float64,Float64}[]
        tracker = propagate(m, algebra, SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6),
                            phase; T=T, nsteps=ceil(Int, T/dt), alpha_history=α_hist)
        return tracker, first.(α_hist), last.(α_hist)
    end

    fmt_T(T) = @sprintf("%.4f", T)

    # ── report ────────────────────────────────────────────────────────────────
    # lines are collected in memory and echoed to stdout as they arrive; the file is written once at the end,
    # and only when save_results is set
    report_lines = String[]
    io_lock = ReentrantLock()
    logline(s) = (lock(io_lock); try
            push!(report_lines, s); println(s)
        finally unlock(io_lock) end)
    logline(@sprintf("# C-Φ control search, k = %d control qubits (n_qubits = %d), Φ = %.4fπ", k, n_qubits, phase/π))
    logline("# generator_fn = $generator_fn, dt = $dt, T window [$tstar_min, $tstar_max], n_random = $n_random, seed = $seed")
    logline(@sprintf("# success_tol = %.0e (a random start counts as a gate), refine_tol = %.0e (threshold of the descent)",
                     success_tol, refine_tol))
    logline(@sprintf("# stall_iters = %d, stall_rel = %.0e (a run stops when J does not improve by more than this for that many iterations)",
                     stall_iters, stall_rel))
    logline("#")
    logline("# ── 1. random starts (in parallel) in batches of $(n_random), until $(target_hits) hit(s) of φ < $(success_tol) or $(max_runs) runs")
    logline("# run   wall/s      T        infidelity   status  gate")

    # ── 1a. warm starts (optional), then random starts in batches of n_random, no polish, until
    #        target_hits hits or max_runs total ──
    Random.seed!(seed)
    solver = SolverParams(tstar_max, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
    nsteps = round(Int, tstar_max/dt)
    one_start() = (randn(d), tstar_min + 0.3 + rand()*(tstar_max - tstar_min - 0.6))
    t0 = time()
    polished = Any[]                                   # one named tuple per run, in order

    # one guarded BFGS from (m0, T0), time weight w; T0 is clamped just inside the window because
    # τ_of_T is only defined on (tstar_min, tstar_max)
    function solve_start(m0, T0)
        t_start = time()
        r1, status = run_bfgs_guarded(algebra, solver, phase, m0, clamp(T0, tstar_min + 1e-3, tstar_max - 1e-3);
                                      time_weight = w, nsteps = nsteps,
                                      iterations = iters_start, g_tol = g_tol,
                                      stall_iters = stall_iters, stall_rel = stall_rel,
                                      time_limit = run_time_limit)
        m1, T1 = minimizer_mT(r1, solver)
        φ1 = Optim.minimum(r1) - w*T1
        return (m = m1, T = T1, φ = φ1, φ_raw = φ1, status = status,
                wall = time() - t_start, promoted = false, warm = false)
    end

    # warm starts: a start derived from the neighbouring phase's solution, so that a phase whose
    # random starts never reach the gate still lands on the monotonic branch instead of a local minimum
    n_warm = min(length(warm_starts), max_runs)
    if n_warm > 0
        println("$n_warm warm start(s) before the random ones")
        warm = Vector{Any}(undef, n_warm)
        Threads.@threads for j in 1:n_warm
            warm[j] = solve_start(warm_starts[j].m, warm_starts[j].T)
        end
        for p in warm
            p = merge(p, (warm = true,))
            push!(polished, p)
            logline(@sprintf("%4d  %9.2f  %9s  %11.3e  %-6s  %s", length(polished), p.wall, fmt_T(p.T),
                             p.φ, p.status, p.φ < success_tol ? "yes" : "warm"))
        end
    end

    while length(polished) < max_runs && count(p -> p.φ < success_tol, polished) < target_hits
        n_done = length(polished)
        n_this = min(n_random, max_runs - n_done)
        println("batch of $n_this random starts  [$n_done done, " *
                "$(count(p -> p.φ < success_tol, polished)) hit(s) of φ < $success_tol, target $target_hits]")
        batch = Vector{Any}(undef, n_this)
        Threads.@threads for j in 1:n_this
            m0, T0 = one_start()
            batch[j] = solve_start(m0, T0)
        end
        for p in batch
            push!(polished, p)
            logline(@sprintf("%4d  %9.2f  %9s  %11.3e  %-6s  %s", length(polished), p.wall, fmt_T(p.T),
                             p.φ, p.status, p.φ < success_tol ? "yes" : ""))
        end
    end
    n_runs = length(polished)

    # ── 1b. promote the n_polish_best lowest-φ runs, and any run already near a gate ──
    for i in sortperm([p.φ for p in polished])[1:min(n_polish_best, n_runs)]
        polished[i] = merge(polished[i], (promoted = true,))
    end
    for i in 1:n_runs
        polished[i].φ <= loose_tol && (polished[i] = merge(polished[i], (promoted = true,)))
    end
    n_promoted = count(p -> p.promoted, polished)

    # ── 1c. polish the promoted runs (w = 0) ──────────────────────────────────
    Threads.@threads for i in 1:n_runs
        polished[i].promoted || continue
        t_start = time()
        r2, status = run_bfgs_guarded(algebra, solver, phase, polished[i].m, polished[i].T; time_weight = 0.0,
                                      nsteps = nsteps, iterations = iters_start, g_tol = g_tol,
                                      stall_iters = stall_iters, stall_rel = stall_rel,
                                      time_limit = run_time_limit)
        m2, T2 = minimizer_mT(r2, solver)
        polished[i] = merge(polished[i], (m = m2, T = T2, φ = Optim.minimum(r2), status = status,
                                          wall = polished[i].wall + time() - t_start))
    end

    logline("")
    logline("# polished runs: raw infidelity → polished (w = 0)")
    logline("# run   wall/s      T        φ raw        φ polished   status")
    for i in 1:n_runs
        polished[i].promoted || continue
        p = polished[i]
        logline(@sprintf("%4d  %9.2f  %9s  %11.3e  %11.3e  %-6s", i, p.wall, fmt_T(p.T), p.φ_raw, p.φ, p.status))
    end

    n_success = count(p -> p.φ < success_tol, polished)
    success   = n_success/n_runs
    n_stall   = count(p -> p.status == "stall", polished)
    n_time    = count(p -> p.status == "time", polished)
    logline("")
    logline(@sprintf("# %d random starts (%d batches, stopped on %s): %d/%d = %.1f%% reached φ < %.0e; %d promoted, %d aborted on stall, %d on the %g s limit   [wall %.1f s, sum of runs %.1f s]",
                     n_runs, cld(n_runs, n_random), n_success >= target_hits ? "the $(target_hits)-hit target" : "max_runs",
                     n_success, n_runs, 100success, success_tol, n_promoted, n_stall, n_time, run_time_limit,
                     time() - t0, sum(p.wall for p in polished)))
    @printf("random starts: %d runs, %d/%d = %.1f%% reached φ < %.0e; %d promoted, %d stalled, %d timed out   [%.1f s]\n",
            n_runs, n_success, n_runs, 100success, success_tol, n_promoted, n_stall, n_time, time() - t0)

    # ── 2. lower T from the shortest distinct gates (in parallel) ─────────────
    gates = sort([p for p in polished if p.φ < success_tol], by = p -> p.T)
    cands = Any[]
    for g in gates
        all(abs(g.T - c.T) > distinct_T for c in cands) && push!(cands, g)
        length(cands) == n_descend && break
    end

    logline("")
    logline("# ── 2. descent: lower T from the $(length(cands)) shortest distinct gates until the gate fails at refine_tol = $refine_tol")
    logline("# start T    T*        infidelity   status  passes")

    descended = Vector{Any}(undef, length(cands))
    m_best = T_best = φ_best = NaN
    t_α, α = Float64[], Float64[]
    if isempty(cands)
        best = polished[argmin([p.φ for p in polished])]
        @printf("no gate found; best φ = %.2e at T = %.4f\n", best.φ, best.T)
        logline(@sprintf("# no gate found (φ < %.0e); best φ = %.2e at T = %.4f", success_tol, best.φ, best.T))
        m_best, T_best, φ_best = best.m, best.T, best.φ
    else
        t_desc = time()
        Threads.@threads for i in eachindex(cands)
            m_i, T_i, status_i = descend(cands[i].m, cands[i].T)
            φ_i = cost_at(m_i, T_i)
            descended[i] = (m = m_i, T = T_i, φ = φ_i, status = status_i)
            logline(@sprintf("%9s  %9s  %11.3e  %-6s  %s", fmt_T(cands[i].T), fmt_T(T_i), φ_i, status_i,
                             φ_i < refine_tol ? "yes" : "no"))
            @printf("  from gate at T = %.4f → T* = %.4f, φ = %.2e\n", cands[i].T, T_i, φ_i)
        end
        # the shortest pulse that meets the threshold (a gate below refine_tol, else the best one found)
        ok = [x for x in descended if x.φ < refine_tol]
        best = isempty(ok) ? descended[argmin([x.φ for x in descended])] : ok[argmin([x.T for x in ok])]
        m_best, T_best, φ_best = best.m, best.T, best.φ

        # a longer polish at the final T if the threshold was missed (BFGS can stall just above it)
        if φ_best >= refine_tol
            m_k, φ_k, _ = fixed_T(m_best, T_best, polish_iters)
            φ_k < φ_best && ((m_best, φ_best) = (m_k, φ_k))
            @printf("  extended polish at T = %.4f (%d iterations) → φ = %.2e\n", T_best, polish_iters, φ_best)
        end
        logline(@sprintf("# descent wall %.1f s", time() - t_desc))

        # control law of the reported pulse
        tracker, t_α, α = control_law(m_best, T_best)
        φ_best = tracker.min_cost
    end

    # ── 3. report ─────────────────────────────────────────────────────────────
    logline("")
    logline("# ── 3. result")
    if !isempty(cands)
        logline(@sprintf("k = %d, Φ = %.4fπ:  T* = %s   infidelity = %.2e   threshold %.0e: %s",
                         k, phase/π, fmt_T(T_best), φ_best, refine_tol, φ_best < refine_tol ? "met" : "NOT met"))
        logline(@sprintf("# success chance of the random starts: %d/%d = %.1f%% (φ < %.0e, T window [%g, %g])",
                         n_success, n_runs, 100success, success_tol, tstar_min, tstar_max))
        logline(@sprintf("# %d α samples on [0, %s], α(0) = %.6f, α(T) - α(0) = %.6f",
                         length(α), fmt_T(T_best), α[1], α[end] - α[1]))
    end

    if save_results
        mkpath(res_dir)
        write(report_file, join(report_lines, "\n") * "\n")
        println("saved ", report_file)
    end

    println("────────────────────────────────────────")
    @printf("k = %d, Φ = %.4fπ:  T* = %s   infidelity = %.2e   success chance = %.1f%% (%d/%d, φ < %.0e)\n",
            k, phase/π, fmt_T(T_best), φ_best, 100success, n_success, n_runs, success_tol)

    p = nothing
    if !isempty(t_α)
        p = plot(t_α, α .- α[1]; lw = 2, xlabel = L"t\,\Omega", ylabel = L"\alpha_{\mathrm{opt}}(t) - \alpha_{\mathrm{opt}}(0)",
                 label = "", title = @sprintf("k = %d control qubits, Φ = %.3gπ:  T* = %s,  infidelity = %.1e",
                                              k, phase/π, fmt_T(T_best), φ_best),
                 titlefontsize = 10, size = plot_size)
        if save_results
            savefig(p, plot_file)
            jldsave(replace(plot_file, ".png" => ".jld2"); phase, k, T = T_best, infidelity = φ_best,
                    success, n_success, n_runs, n_random, target_hits, max_runs, success_tol, refine_tol, m = m_best, t = t_α, alpha = α,
                    φ_starts = [x.φ for x in polished], φ_raw = [x.φ_raw for x in polished],
                    T_starts = [x.T for x in polished], promoted = [x.promoted for x in polished],
                    warm = [x.warm for x in polished],
                    status = [x.status for x in polished], n_promoted, stall_iters, stall_rel, run_time_limit)
            println("saved ", plot_file, " and ", replace(plot_file, ".png" => ".jld2"))
        end
    end

    return (phase = phase, k = k, T = T_best, φ = φ_best, success = success, n_success = n_success,
            n_runs = n_runs, n_random = n_random, target_hits = target_hits, max_runs = max_runs,
            success_tol = success_tol, refine_tol = refine_tol,
            m = m_best, t = t_α, α = α, polished = polished, plot = p)
end
