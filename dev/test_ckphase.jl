# src is included only once, so that this file can be included repeatedly (e.g. by dev/sweep_ckphase.jl)
if !@isdefined(Algebra)
    include("../src/structs.jl")
    include("../src/propagation.jl")
    include("../src/optimise.jl")
end

using Plots, LaTeXStrings, Random, Printf, JLD2, LinearAlgebra

# Minimal gate time T* of the controlled-phase gate C-Φ = diag(1, …, 1, e^{iΦ}) for ONE phase Φ:
#   1. n_random random starts in parallel (BFGS on J = φ + w T, then a polish with w = 0); one line per run
#      (time, T, infidelity) is written to the report file as soon as the run finishes
#   2. from the n_descend shortest distinct gates, T is lowered at fixed T (step + bisection) with the
#      stricter threshold refine_tol, so that the reported T* is resolved to ~T_step/2^n_bisect
#   3. the shortest pulse meeting refine_tol is reported (txt) and its control α_opt(t) plotted (png)
# Output: res/ckphase_k<k>_phase_<Φ/π>π_control.txt and _control.png
# Run:  julia -t auto --project=. dev/test_ckphase.jl

# inputs (an including script may set any of them before this file is included)
n_qubits = 5
phase = float(pi)
n_random = 96
tstar_min = 15.0
tstar_max = 25.0

# ── deeper settings ───────────────────────────────────────────────────────────
dt = 1e-2
generator_fn = construct_HJB_generators
seed        = 1
success_tol = 1e-8      # a random start counts as successful if its polished infidelity is below this
loose_frac  = 0.1       # first stage: a start is polished only if φ < min(1e-2, loose_frac * φ_id)
refine_tol  = 1e-10     # threshold the descending pulse must meet (an accuracy check on T*)
n_descend   = 4         # number of shortest distinct gates from which T is lowered
distinct_T  = 0.02      # two gates are distinct if their T differ by more than this
time_weight = 1e-3      # time penalty of the first stage at Φ = π (scaled with Φ below)
T_step      = 0.1       # step by which T is lowered before bisection
n_bisect    = 11        # final T resolution T_step / 2^n_bisect ≈ 5e-5
tstar_floor = 0.5       # the descent never reports a pulse shorter than this (tstar_min bounds the starts only)
reltol = abstol = dist_tol = grad_tol = unitary_tol = 1e-8
iters_start = 300       # BFGS iterations of the polish of a random start
iters_fixed  = 300      # BFGS iterations of every fixed-T step of the descent
polish_iters = 1000     # BFGS iterations of the retry/polish when a descent step misses refine_tol
g_tol       = 1e-6      # the analytic gradient matches the discretised cost only to O(dt²)
plot_size   = (600, 380)

# ── setup ─────────────────────────────────────────────────────────────────────
k = n_qubits - 1
im_control, im_drift = im .* generator_fn(n_qubits)
algebra = Algebra(im_control, im_drift)
d = length(algebra.lie_basis)

# infidelity of doing nothing (U = I) for C-Φ: 1 - cos(Φ/4). The time weight and the first-stage acceptance
# threshold are taken relative to it, so that at small Φ the identity neither beats a gate on J nor passes as a candidate
φ_id      = 1 - cos(phase/4)
w         = time_weight * φ_id/(1 - cos(π/4))
loose_tol = min(1e-2, loose_frac*φ_id)

res_dir = joinpath(@__DIR__, "..", "res")
tag = @sprintf("ckphase_k%d_phase_%gpi", k, round(phase/π, digits=4))
report_file = joinpath(res_dir, tag * "_report.txt")
plot_file   = joinpath(res_dir, tag * "_control.png")

println("Setup finished: k = $k, Φ = $(round(phase/π, digits=4))π, $(Threads.nthreads()) threads, d = $d")

# ── helpers ───────────────────────────────────────────────────────────────────
# optimise m at fixed T (window T ± 1e-6, so the time weight is irrelevant); returns (m, infidelity)
function fixed_T(m, T, iters = iters_fixed)
    s = SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6)
    r = run_bfgs(m, T, algebra, s, phase; time_weight=0.0, nsteps=ceil(Int, T/dt), iterations=iters, g_tol=g_tol)
    return minimizer_mT(r, s)[1], Optim.minimum(r)
end

# infidelity of (m, T) on the T/dt grid used by fixed_T
cost_at(m, T) = propagate(m, algebra, SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6),
                          phase; T=T, nsteps=ceil(Int, T/dt)).min_cost

# lower T from a gate (m, T) while the gate still meets refine_tol, then bisect between the last gate and the first failure.
# A step that misses refine_tol is retried with a longer polish before it counts as a failure: from a warm start the
# fixed-T solve is a polish that should reach ~1e-12, and a failure is only meaningful once it has really converged.
function descend(m, T)
    T_ok, T_bad = T, NaN
    step_failed(m_k, T_try) = begin
        m_k, φ = fixed_T(m, T_try)
        φ < refine_tol && return (true, m_k)
        m_k, φ = fixed_T(m, T_try, polish_iters)
        return (φ < refine_tol, m_k)
    end
    while T_ok - T_step > tstar_floor
        ok, m_k = step_failed(m, T_ok - T_step)
        ok || (T_bad = T_ok - T_step; break)
        m, T_ok = m_k, T_ok - T_step
    end
    isnan(T_bad) && return m, T_ok
    for _ in 1:n_bisect
        T_mid = (T_ok + T_bad)/2
        ok, m_k = step_failed(m, T_mid)
        ok ? ((m, T_ok) = (m_k, T_mid)) : (T_bad = T_mid)
    end
    return m, T_ok
end

# evolution with the found control, from the Lie costate: (tracker, t, α)
function control_law(m, T)
    α_hist = Tuple{Float64,Float64}[]
    tracker = propagate(m, algebra, SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6),
                        phase; T=T, nsteps=ceil(Int, T/dt), alpha_history=α_hist)
    return tracker, first.(α_hist), last.(α_hist)
end

fmt_T(T) = @sprintf("%.4f", T)

# ── report ────────────────────────────────────────────────────────────────────
io_lock = ReentrantLock()
logline(io, s) = (lock(io_lock); try println(io, s); flush(io); finally unlock(io_lock) end)

io = open(report_file, "w")
println(io, "# C-Φ control search, k = $k control qubits (n_qubits = $n_qubits), Φ = $(round(phase/π, digits=4))π")
println(io, "# generator_fn = $generator_fn, dt = $dt, T window [$tstar_min, $tstar_max], n_random = $n_random, seed = $seed")
println(io, "# success_tol = $success_tol (a random start counts as a gate), refine_tol = $refine_tol (threshold of the descent)")
println(io)
println(io, "# ── 1. random starts (in parallel), per run: wall time, final time, infidelity")
println(io, "# run   wall/s      T        infidelity   gate")
flush(io)

# ── 1. random starts ──────────────────────────────────────────────────────────
Random.seed!(seed)
solver = SolverParams(tstar_max, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
nsteps = round(Int, tstar_max/dt)
starts = [(randn(d), tstar_min + 0.3 + rand()*(tstar_max - tstar_min - 0.6)) for _ in 1:n_random]
polished = Vector{Any}(undef, n_random)
wall      = Vector{Float64}(undef, n_random)
t0 = time()
Threads.@threads for i in 1:n_random
    m0, T0 = starts[i]
    t_start = time()
    r1 = run_bfgs(m0, T0, algebra, solver, phase; time_weight=w, nsteps=nsteps)
    m1, T1 = minimizer_mT(r1, solver)
    φ1 = Optim.minimum(r1) - w*T1
    if φ1 > loose_tol                       # not close to a gate: not polished
        polished[i] = (m = m1, T = T1, φ = φ1)
    else
        r2 = run_bfgs(m1, T1, algebra, solver, phase; time_weight=0.0, nsteps=nsteps, iterations=iters_start, g_tol=g_tol)
        m2, T2 = minimizer_mT(r2, solver)
        polished[i] = (m = m2, T = T2, φ = Optim.minimum(r2))
    end
    wall[i] = time() - t_start
    logline(io, @sprintf("%4d  %9.2f  %9s  %11.3e  %s", i, wall[i], fmt_T(polished[i].T), polished[i].φ,
                         polished[i].φ < success_tol ? "yes" : "no"))
    @printf("start %3d: φ = %.2e at T = %.4f   [%.1f s]\n", i, polished[i].φ, polished[i].T, wall[i])
end
n_success = count(p -> p.φ < success_tol, polished)
success   = n_success/n_random
logline(io, "")
logline(io, @sprintf("# random starts: %d/%d = %.1f%% reached φ < %.0e   [wall %.1f s, sum of runs %.1f s]",
                      n_success, n_random, 100success, success_tol, time() - t0, sum(wall)))
@printf("random starts: %d/%d = %.1f%% reached φ < %.0e   [%.1f s]\n", n_success, n_random, 100success, success_tol, time() - t0)

# ── 2. lower T from the shortest distinct gates (in parallel) ──────────────────
gates = sort([p for p in polished if p.φ < success_tol], by = p -> p.T)
cands = Any[]
for g in gates
    all(abs(g.T - c.T) > distinct_T for c in cands) && push!(cands, g)
    length(cands) == n_descend && break
end

logline(io, "")
logline(io, "# ── 2. descent: lower T from the $(length(cands)) shortest distinct gates until the gate fails at refine_tol = $refine_tol")
logline(io, "# start T    T*        infidelity   passes refine_tol")

descended = Vector{Any}(undef, length(cands))
m_best = T_best = φ_best = NaN
t_α, α = Float64[], Float64[]
if isempty(cands)
    best = polished[argmin([p.φ for p in polished])]
    @printf("no gate found; best φ = %.2e at T = %.4f\n", best.φ, best.T)
    logline(io, @sprintf("# no gate found (φ < %.0e); best φ = %.2e at T = %.4f", success_tol, best.φ, best.T))
    m_best, T_best, φ_best = best.m, best.T, best.φ
else
    t_desc = time()
    Threads.@threads for i in eachindex(cands)
        m_i, T_i = descend(cands[i].m, cands[i].T)
        φ_i = cost_at(m_i, T_i)
        descended[i] = (m = m_i, T = T_i, φ = φ_i)
        logline(io, @sprintf("%9s  %9s  %11.3e  %s", fmt_T(cands[i].T), fmt_T(T_i), φ_i,
                             φ_i < refine_tol ? "yes" : "no"))
        @printf("  from gate at T = %.4f → T* = %.4f, φ = %.2e\n", cands[i].T, T_i, φ_i)
    end
    # the shortest pulse that meets the threshold (a gate below refine_tol, else the best one found)
    ok = [x for x in descended if x.φ < refine_tol]
    best = isempty(ok) ? descended[argmin([x.φ for x in descended])] : ok[argmin([x.T for x in ok])]
    m_best, T_best, φ_best = best.m, best.T, best.φ

    # a longer polish at the final T if the threshold was missed (BFGS can stall just above it)
    if φ_best >= refine_tol
        m_k, φ_k = fixed_T(m_best, T_best, polish_iters)
        φ_k < φ_best && ((m_best, φ_best) = (m_k, φ_k))
        @printf("  extended polish at T = %.4f (%d iterations) → φ = %.2e\n", T_best, polish_iters, φ_best)
    end
    logline(io, @sprintf("# descent wall %.1f s", time() - t_desc))

    # control law of the reported pulse
    tracker, t_α, α = control_law(m_best, T_best)
    φ_best = tracker.min_cost
end

# ── 3. report ─────────────────────────────────────────────────────────────────
logline(io, "")
logline(io, "# ── 3. result")
if !isempty(cands)
    logline(io, @sprintf("k = %d, Φ = %.4fπ:  T* = %s   infidelity = %.2e   threshold %.0e: %s",
                         k, phase/π, fmt_T(T_best), φ_best, refine_tol, φ_best < refine_tol ? "met" : "NOT met"))
    logline(io, @sprintf("# success chance of the random starts: %d/%d = %.1f%% (φ < %.0e, T window [%g, %g])",
                         n_success, n_random, 100success, success_tol, tstar_min, tstar_max))
    logline(io, @sprintf("# %d α samples on [0, %s], α(0) = %.6f, α(T) - α(0) = %.6f",
                         length(α), fmt_T(T_best), α[1], α[end] - α[1]))
end
close(io)
println("saved ", report_file)

println("────────────────────────────────────────")
@printf("k = %d, Φ = %.4fπ:  T* = %s   infidelity = %.2e   success chance = %.1f%% (%d/%d, φ < %.0e)\n",
        k, phase/π, fmt_T(T_best), φ_best, 100success, n_success, n_random, success_tol)

p = nothing
if !isempty(t_α)
    p = plot(t_α, α .- α[1]; lw = 2, xlabel = L"t\,\Omega", ylabel = L"\alpha_{\mathrm{opt}}(t) - \alpha_{\mathrm{opt}}(0)",
             label = "", title = @sprintf("k = %d control qubits, Φ = %.3gπ:  T* = %s,  infidelity = %.1e",
                                          k, phase/π, fmt_T(T_best), φ_best),
             titlefontsize = 10, size = plot_size)
    savefig(p, plot_file)
    jldsave(replace(plot_file, ".png" => ".jld2"); phase, k, T = T_best, infidelity = φ_best,
            success, n_success, n_random, success_tol, refine_tol, m = m_best, t = t_α, alpha = α,
            φ_starts = [x.φ for x in polished], T_starts = [x.T for x in polished], wall)
    println("saved ", plot_file, " and ", replace(plot_file, ".png" => ".jld2"))
end
p
