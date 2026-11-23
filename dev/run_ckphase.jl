# Minimal gate time T* and optimal control α_opt(t) of the controlled-phase gate C-Φ for ONE phase Φ.
# The search itself lives in dev/ckphase.jl (run_ckphase); this file only sets the inputs.
# Output (save_results = true): res/c_<k>_phase_<Φ/π>π_report.txt, _control.png, _control.jld2
# Run:  julia -t auto --project=. dev/run_ckphase.jl

include(joinpath(@__DIR__, "ckphase.jl"))

# inputs
n_qubits = 2                # control qubits k = n_qubits - 1
phase = float(pi)           # the controlled phase Φ of C-Φ
n_random = 48               # random starts per batch
success_tol = 1e-4          # a run counts as a hit (and satisfies the batch loop) below this infidelity
target_hits = 1             # keep launching batches until this many runs reach φ < success_tol …
max_runs = 5*n_random               # … or this many runs have been launched in total
run_time_limit = 300.0      # abandon a single run after this many seconds
stall_iters = 40            # abandon a single run after this many iterations without progress
tstar_min = 5.0            # random starts draw their initial T in [tstar_min, tstar_max]
tstar_max = 20.0
generator_fn = construct_HJB_generators
save_results = true        # write the report, the control plot and the .jld2?

res = run_ckphase(; phase, n_random, n_qubits, tstar_min, tstar_max, generator_fn, save_results,
                    success_tol, target_hits, max_runs, run_time_limit, stall_iters)

@printf("k = %d, Φ = %.4fπ:  T* = %.4f   infidelity = %.2e\n", res.k, res.phase/π, res.T, res.φ)
res.plot
