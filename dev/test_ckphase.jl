# src is included only once, so that this file can be included repeatedly (e.g. by dev/sweep_ckphase.jl)
if !@isdefined(Algebra)
    include("../src/structs.jl")
    include("../src/propagation.jl")
    include("../src/optimise.jl")
end

using Plots, LaTeXStrings, Random, Printf, JLD2

# Minimal gate time T* of the controlled-phase gate C-Φ = diag(1, 1, 1, e^{iΦ}) (up to single-qubit phases) for ONE phase Φ:
#   1. random starts (in parallel): BFGS on J = φ + w T, then a polish with w = 0  → success chance at success_tol
#   2. from the shortest distinct gates, lower T at fixed T (step + bisection) → T*
#   3. output T*, infidelity, success chance and the control law α_opt(t) of the best gate
# Same procedure as dev/plot_T_vs_controlled_phase.jl, for a single Φ.

# ── settings ──────────────────────────────────────────────────────────────────
# phase, n_random and generator_fn keep their values if they are already defined (set by a sweep script before include)
@isdefined(phase)        || (phase = π)                                # target controlled phase Φ
@isdefined(n_random)     || (n_random = 16)                            # number of random starts
@isdefined(generator_fn) || (generator_fn = construct_HJB_generators)   # returns (control, drift)
success_tol = 1e-8     # a start is successful if its polished infidelity is below this
descend_tol = 1e-8     # infidelity that still counts as a gate while T is lowered
n_descend   = 4        # number of shortest distinct gates from which T is lowered
time_weight = 1e-3     # time penalty at Φ = π (scaled with Φ below)
T_step      = 0.1      # step by which T is lowered before bisection
n_bisect    = 11       # final T resolution T_step / 2^n_bisect ≈ 5e-5
tstar_min   = 2.5      # T window of the random starts (excludes the T → 0 identity trap)
tstar_max   = 20.0
dt          = 1e-2
seed        = 1

# ── setup ─────────────────────────────────────────────────────────────────────
n_qubits  = 3
gate_type = "k1"
im_control, im_drift = im .* generator_fn(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = abstol = dist_tol = grad_tol = unitary_tol = 1e-8
d   = length(algebra.lie_basis)
tar = build_tar(gate_type, phase, n_qubits, algebra)
println("Setup finished: Φ = $(round(phase/π, digits=4))π, $(Threads.nthreads()) threads")

# infidelity of doing nothing (U = I) for C-Φ; time weight and first-stage threshold are taken relative to it
φ_id      = 1 - cos(phase/4)
w         = time_weight * φ_id/(1 - cos(π/4))
loose_tol = min(1e-2, 0.1*φ_id)

# optimise m at fixed T (T window T ± 1e-6); returns (m, infidelity)
function fixed_T(m, T)
    s = SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6)
    r = run_bfgs(m, T, algebra, s, tar; time_weight=0.0, nsteps=ceil(Int, T/dt), iterations=300, g_tol=1e-6)
    return minimizer_mT(r, s)[1], Optim.minimum(r)
end

# lower T from a gate (m, T) while a gate still exists, then bisect between the last gate and the first failure
function descend(m, T)
    T_ok, T_bad = T, NaN
    while T_ok - T_step > tstar_min
        m_k, φ = fixed_T(m, T_ok - T_step)
        φ < descend_tol || (T_bad = T_ok - T_step; break)
        m, T_ok = m_k, T_ok - T_step
    end
    isnan(T_bad) && return m, T_ok
    for _ in 1:n_bisect
        T_mid = (T_ok + T_bad)/2
        m_k, φ = fixed_T(m, T_mid)
        φ < descend_tol ? ((m, T_ok) = (m_k, T_mid)) : (T_bad = T_mid)
    end
    return m, T_ok
end

# ── 1. random starts ──────────────────────────────────────────────────────────
Random.seed!(seed)
solver = SolverParams(tstar_max, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
nsteps = round(Int, tstar_max/dt)
starts = [(randn(d), tstar_min + 0.3 + rand()*(tstar_max - tstar_min - 0.6)) for _ in 1:n_random]
polished = Vector{Any}(undef, n_random)
t0 = time()
Threads.@threads for k in 1:n_random
    m0, T0 = starts[k]
    r1 = run_bfgs(m0, T0, algebra, solver, tar; time_weight=w, nsteps=nsteps)
    m1, T1 = minimizer_mT(r1, solver)
    φ1 = Optim.minimum(r1) - w*T1
    if φ1 > loose_tol                       # not close to a gate: not polished
        polished[k] = (m = m1, T = T1, φ = φ1)
    else
        r2 = run_bfgs(m1, T1, algebra, solver, tar; time_weight=0.0, nsteps=nsteps, iterations=300)
        m2, T2 = minimizer_mT(r2, solver)
        polished[k] = (m = m2, T = T2, φ = Optim.minimum(r2))
    end
    @printf("start %3d: φ = %.2e at T = %.4f\n", k, polished[k].φ, polished[k].T)
end
n_success = count(p -> p.φ < success_tol, polished)
success   = n_success/n_random
@printf("random starts: %d/%d = %.1f%% reached φ < %.0e   [%d s]\n", n_success, n_random, 100success, success_tol, round(Int, time() - t0))

# ── 2. lower T from the shortest distinct gates ───────────────────────────────
gates = sort([p for p in polished if p.φ < success_tol], by = p -> p.T)
cands = Any[]
for g in gates
    all(abs(g.T - c.T) > 0.02 for c in cands) && push!(cands, g)
    length(cands) == n_descend && break
end
if isempty(cands)
    best = polished[argmin([p.φ for p in polished])]
    @printf("no gate found; best φ = %.2e at T = %.4f\n", best.φ, best.T)
    m_best, T_best = best.m, best.T
else
    descended = Vector{Any}(undef, length(cands))
    Threads.@threads for k in eachindex(cands)
        descended[k] = descend(cands[k].m, cands[k].T)
        @printf("  from gate at T = %.4f → T* = %.4f\n", cands[k].T, descended[k][2])
    end
    m_best, T_best = descended[argmin([x[2] for x in descended])]
end

# ── 3. result and control law α_opt(t) ────────────────────────────────────────
s_best = SolverParams(T_best, reltol, abstol, dist_tol, grad_tol, unitary_tol, T_best - 1e-6, T_best + 1e-6)
alpha_hist = Tuple{Float64,Float64}[]
tracker = new_propagate(m_best, algebra, s_best, tar; T=T_best, nsteps=ceil(Int, T_best/dt), alpha_history=alpha_hist)
t_α, α = first.(alpha_hist), last.(alpha_hist)
φ_best = tracker.min_cost

println("────────────────────────────────────────")
@printf("Φ = %.4fπ:  T* = %.4f   infidelity = %.2e   success chance = %.1f%% (%d/%d, φ < %.0e)\n",
        phase/π, T_best, φ_best, 100success, n_success, n_random, success_tol)

p = plot(t_α, α .- α[1]; lw = 2, label = @sprintf("Φ = %.3gπ, T* = %.3f", phase/π, T_best),
         xlabel = L"t\,\Omega", ylabel = L"\alpha_{\mathrm{opt}}(t) - \alpha_{\mathrm{opt}}(0)", size = (600, 380))
# out = joinpath(@__DIR__, "..", "res", @sprintf("single_phase_%.4fpi", phase/π))
# savefig(p, out * "_alpha.png")
# jldsave(out * ".jld2"; phase, T = T_best, infidelity = φ_best, success, n_success, n_random, success_tol,
#         m = m_best, t = t_α, alpha = α, φ_starts = [p.φ for p in polished], T_starts = [p.T for p in polished])
# println("saved ", out, "_alpha.png and .jld2")
p
