include("../src/structs.jl")
include("../src/propagation.jl")
include("../src/optimise.jl")

using Plots, LaTeXStrings, Random, Printf, JLD2, Logging

# minimal gate time T* of the controlled-phase gate C-Φ against Φ (cf. Fromonteil et al., PRR 6, 033333 (2024), Fig. 3d)
# cost: logical subspace, CZ-type target up to single-qubit phases (θ in infidelity)
# run with threads, e.g.  julia -t auto --project=. dev/plot_T_vs_controlled_phase.jl

# generators
n_qubits = 2
gate_type = "k1"
dt = 1e-2

im_control, im_drift = im .* construct_HJB_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
d = length(algebra.lie_basis)
println("Setup finished, ", Threads.nthreads(), " threads")
##
Φs          = [collect(range(π, 0.05π, step=-0.05π)); 0.025π]   # swept from π downwards
n_random    = 96       # random starts per Φ (run in parallel), besides the continuation from the previous Φ
n_descend   = 4        # number of shortest distinct candidate gates from which T is lowered
time_weight = 1e-3     # first stage of a random start at Φ = π: J = φ + w*T (w is scaled with Φ, see below)
gate_tol    = 1e-6     # infidelity below which a result counts as a gate
descend_tol = 1e-8     # stricter threshold while lowering T: just below T* the infidelity is still < 1e-6 (T*(π) would come out 7.605)
success_tol = 1e-8     # a random start counts as successful if its polished infidelity is below this
T_step      = 0.1      # step by which T is lowered before bisection
n_bisect    = 11       # final T resolution T_step / 2^n_bisect ≈ 5e-5
tstar_min   = 2.5      # T window of the random starts; excludes the T → 0 identity trap (paper: T* > 3 for all Φ)
tstar_max   = 10.0
n_refine    = 3        # passes in which every Φ is retried from the solutions of its neighbours

# infidelity of "doing nothing" (U = I) for the target C-Φ, maximised over θ: 1 - cos(Φ/4).
# It sets the scale of the cost: the time weight and the first-stage acceptance threshold are taken relative to it,
# so that at small Φ the identity neither beats a gate on J nor passes as a candidate.
φ_id(Φ)      = 1 - cos(Φ/4)
w_of(Φ)      = time_weight * φ_id(Φ)/φ_id(π)
loose_tol(Φ) = min(1e-2, 0.1*φ_id(Φ))

# optimise m at fixed T (T window T ± 1e-6); returns (m, infidelity)
function fixed_T(m, T, tar)
    s = SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6)
    r = run_bfgs(m, T, algebra, s, tar; time_weight=0.0, nsteps=ceil(Int, T/dt), iterations=300, g_tol=1e-6)
    return minimizer_mT(r, s)[1], Optim.minimum(r)
end

# infidelity of (m, T), propagated on the same grid as fixed_T
function cost_at(m, T, tar)
    s = SolverParams(T, reltol, abstol, dist_tol, grad_tol, unitary_tol, T - 1e-6, T + 1e-6)
    return propagate(m, algebra, s, tar; T=T, nsteps=ceil(Int, T/dt)).min_cost
end

# lower T from a gate (m, T) while a gate still exists, then bisect between the last gate and the first failure
function descend(m, T, tar)
    T_ok = T
    T_bad = NaN
    while T_ok - T_step > tstar_min
        m_k, φ = fixed_T(m, T_ok - T_step, tar)
        if φ < descend_tol
            m, T_ok = m_k, T_ok - T_step
        else
            T_bad = T_ok - T_step
            break
        end
    end
    isnan(T_bad) && return m, T_ok
    for _ in 1:n_bisect
        T_mid = (T_ok + T_bad)/2
        m_k, φ = fixed_T(m, T_mid, tar)
        φ < descend_tol ? ((m, T_ok) = (m_k, T_mid)) : (T_bad = T_mid)
    end
    return m, T_ok
end

# a gate from a warm start (m, T): try T, T + T_step, ... until a gate is found, then descend
function from_warm_start(m, T, tar; n_up=5)
    for k in 0:n_up
        m_k, φ = fixed_T(m, T + k*T_step, tar)
        φ < descend_tol && return descend(m_k, T + k*T_step, tar)
    end
    return nothing
end

# random starts: stage 1 with time penalty, stage 2 polish without; returns all polished results
function random_starts(Φ, tar, starts)
    solver = SolverParams(tstar_max, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
    nsteps = round(Int, tstar_max/dt)
    out = Vector{Any}(nothing, length(starts))
    Threads.@threads for k in eachindex(starts)
        m0, T0 = starts[k]
        r1 = run_bfgs(m0, T0, algebra, solver, tar; time_weight=w_of(Φ), nsteps=nsteps)
        m1, T1 = minimizer_mT(r1, solver)
        φ1 = Optim.minimum(r1) - w_of(Φ)*T1
        if φ1 > loose_tol(Φ)
            out[k] = (m = m1, T = T1, φ = φ1)
            continue
        end
        r2 = run_bfgs(m1, T1, algebra, solver, tar; time_weight=0.0, nsteps=nsteps, iterations=300)
        m2, T2 = minimizer_mT(r2, solver)
        out[k] = (m = m2, T = T2, φ = Optim.minimum(r2))
    end
    return out
end

function sweep(Φs)
    prev = nothing
    results = Vector{Any}(nothing, length(Φs))
    for (i, Φ) in enumerate(Φs)
        t0 = time()
        tar = build_tar(gate_type, Φ, n_qubits, algebra)
        # starting points drawn on the main thread, so that the run is reproducible with threads
        starts = [(randn(d), tstar_min + 0.3 + rand()*(tstar_max - tstar_min - 0.6)) for _ in 1:n_random]
        polished = random_starts(Φ, tar, starts)
        gates = sort([p for p in polished if p.φ < gate_tol], by = p -> p.T)
        φ_starts  = [p.φ for p in polished]    # infidelity reached by every random start (stage-1 value if it was not polished)
        T_starts  = [p.T for p in polished]
        n_success = count(<(success_tol), φ_starts)
        # continuation: the previous (larger) Φ has the longer T*, so its gate is a warm start above the current T*
        candidates = Any[]
        prev === nothing || push!(candidates, (m = prev.m, T = prev.T, warm = true))
        for g in gates   # shortest distinct gate times
            all(abs(g.T - c.T) > 0.02 for c in candidates if !c.warm) && push!(candidates, (m = g.m, T = g.T, warm = false))
            count(c -> !c.warm, candidates) == n_descend && break
        end
        descended = Vector{Any}(nothing, length(candidates))
        Threads.@threads for k in eachindex(candidates)
            c = candidates[k]
            descended[k] = c.warm ? from_warm_start(c.m, c.T, tar) : descend(c.m, c.T, tar)
        end
        descended = filter(!isnothing, descended)
        if isempty(descended)
            best = polished[argmin([p.φ for p in polished])]     # no gate: keep the lowest infidelity reached
            results[i] = (Φ = Φ, T = best.T, m = best.m, φ = best.φ, gate = false, success = n_success/n_random, φ_starts = φ_starts, T_starts = T_starts)
            @printf("Φ = %.3f π: no gate (best φ = %.2e at T = %.4f)  success %d/%d = %.1f%%   [%d s]\n",
                    Φ/π, best.φ, best.T, n_success, n_random, 100*n_success/n_random, round(Int, time() - t0))
        else
            m, T = descended[argmin([x[2] for x in descended])]
            # success = fraction of random starts whose polished infidelity is below success_tol (before descending in T)
            results[i] = (Φ = Φ, T = T, m = m, φ = cost_at(m, T, tar), gate = true, success = n_success/n_random, φ_starts = φ_starts, T_starts = T_starts)
            prev = results[i]
            @printf("Φ = %.3f π: T* = %.4f  φ = %.1e  success (φ < %.0e) %d/%d = %.1f%%   [%d s]\n",
                    Φ/π, T, results[i].φ, success_tol, n_success, n_random, 100*n_success/n_random, round(Int, time() - t0))
        end
        flush(stdout)
    end
    return results
end

# retry every Φ from the gates of its two neighbours; keep a shorter gate if one is found
function refine!(results)
    improved = false
    for i in eachindex(results)
        Φ = results[i].Φ
        tar = build_tar(gate_type, Φ, n_qubits, algebra)
        nbrs = [results[j] for j in (i-1, i+1) if checkbounds(Bool, results, j) && results[j].gate]
        found = Vector{Any}(nothing, length(nbrs))
        Threads.@threads for k in eachindex(nbrs)
            found[k] = from_warm_start(nbrs[k].m, nbrs[k].T, tar)
        end
        for f in found
            f === nothing && continue
            m, T = f
            if !results[i].gate || T < results[i].T - 1e-4
                @printf("  refine Φ = %.3f π: T* %.4f → %.4f\n", Φ/π, results[i].gate ? results[i].T : NaN, T)
                results[i] = merge(results[i], (T = T, m = m, φ = cost_at(m, T, tar), gate = true))   # keeps the random-start success rate
                improved = true
            end
        end
    end
    return improved
end

Random.seed!(1)
global_logger(ConsoleLogger(stderr, Logging.Error))   # silence ADM / solver warnings during the sweep
@time results = sweep(Φs)
for pass in 1:n_refine
    println("refinement pass $pass"); flush(stdout)
    refine!(results) || break
end
global_logger(ConsoleLogger(stderr, Logging.Info))

# ── control profiles α_opt(t) of selected gates ───────────────────────────────
function alpha_profile(r)
    tar = build_tar(gate_type, r.Φ, n_qubits, algebra)
    s = SolverParams(r.T, reltol, abstol, dist_tol, grad_tol, unitary_tol, r.T - 1e-6, r.T + 1e-6)
    hist = Tuple{Float64,Float64}[]
    propagate(r.m, algebra, s, tar; T=r.T, nsteps=ceil(Int, r.T/dt), alpha_history=hist)
    return first.(hist), last.(hist)
end
Φ_show = [π, 0.8π, 0.6π, 0.4π, 0.2π]
gates  = [r for r in results if r.gate && r.φ <= gate_tol]
shown  = [gates[argmin([abs(g.Φ - Φ) for g in gates])] for Φ in Φ_show if any(abs(g.Φ - Φ) < 1e-8 for g in gates)]
lowest = gates[argmin([g.Φ for g in gates])]
lowest in shown || push!(shown, lowest)       # also the smallest phase for which a gate was found
profiles = [(r, alpha_profile(r)...) for r in shown]

# ── plots ─────────────────────────────────────────────────────────────────────
Φ_res = [r.Φ for r in results]
T_res = [r.T for r in results]
logφ  = [log10(max(r.φ, 1e-16)) for r in results]
isg   = [r.gate for r in results]
clims = (minimum(logφ), max(0.0, maximum(logφ)))
p = scatter(Φ_res[isg] ./ π, T_res[isg]; marker_z = logφ[isg], clims = clims, color = :viridis, markersize = 5,
            markerstrokewidth = 0, label = "gate (φ ≤ $(gate_tol))", colorbar_title = L"\log_{10}\,\varphi",
            xlims = (0, 1.03), ylims = (0, 10), xlabel = L"\Phi/\pi", ylabel = L"T^*\Omega", legend = :bottomright, size = (560, 400))
any(.!isg) && scatter!(p, Φ_res[.!isg] ./ π, T_res[.!isg]; marker_z = logφ[.!isg], clims = clims, color = :viridis,
                       marker = :xcross, markersize = 6, label = "no gate: best φ reached")
scatter!(p, [1.0], [7.611]; marker = :star5, color = :red, markersize = 7, label = "paper: CZ, 7.611")

q = plot(xlabel = L"t\,\Omega", ylabel = L"\alpha_{\mathrm{opt}}(t) - \alpha_{\mathrm{opt}}(0)", legend = :outerright, size = (700, 400))
for (r, ts, αs) in profiles
    plot!(q, ts, αs .- αs[1]; lw = 2, label = @sprintf("Φ = %.3gπ, T* = %.3f", r.Φ/π, r.T))
end

# same T*(Φ), coloured by the percentage of random starts that reached a gate
succ = [100*r.success for r in results]
p_s = scatter(Φ_res ./ π, T_res; marker_z = succ, clims = (0, 100), color = :viridis, markersize = 6, markerstrokewidth = 0,
              marker = [g ? :circle : :xcross for g in isg], label = "",
              colorbar_title = "successful random starts (%), φ < $(success_tol)",
              xlims = (0, 1.03), ylims = (0, 10), xlabel = L"\Phi/\pi", ylabel = L"T^*\Omega", size = (600, 400), right_margin = 5Plots.mm)
scatter!(p_s, [1.0], [7.611]; marker = :star5, color = :red, markersize = 7, label = "paper: CZ, 7.611", legend = :bottomright)

savefig(p, joinpath(@__DIR__, "..", "res", "T_vs_controlled_phase.png"))
savefig(p_s, joinpath(@__DIR__, "..", "res", "T_vs_controlled_phase_success.png"))
savefig(q, joinpath(@__DIR__, "..", "res", "alpha_controls.png"))
jldsave(joinpath(@__DIR__, "..", "res", "T_vs_controlled_phase.jld2");
        Φ = Φ_res, T = T_res, φ = [r.φ for r in results], gate = isg, success = [r.success for r in results], n_random = n_random,
        success_tol = success_tol, φ_starts = reduce(hcat, [r.φ_starts for r in results]), T_starts = reduce(hcat, [r.T_starts for r in results]),
        m = reduce(hcat, [r.m for r in results]),
        alpha_Φ = [r.Φ for (r, _, _) in profiles], alpha_t = [ts for (_, ts, _) in profiles], alpha = [αs for (_, _, αs) in profiles])
open(joinpath(@__DIR__, "..", "res", "T_vs_controlled_phase.txt"), "w") do io
    println(io, "# Φ/π   T*Ω   infidelity   gate   success rate of $(n_random) random starts (%, polished φ < $(success_tol))")
    for r in results
        @printf(io, "%.4f  %.4f  %.2e  %-5s  %.1f\n", r.Φ/π, r.T, r.φ, r.gate, 100*r.success)
    end
end
display(p); display(q)
