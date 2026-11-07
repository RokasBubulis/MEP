include("../src/structs.jl")
include("../src/propagation.jl")
include("../src/optimise.jl")

using Plots, LaTeXStrings, Random, Printf, JLD2

# setup (as in test_bfgs_multistart.jl)
n_qubits = 2
gate_type = "k1"
dt=1e-2

im_control, im_drift = im .* construct_HJB_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-8
d = length(algebra.lie_basis)
println("Setup finished")
##
# minimal gate time T* against the target phase θ of the controlled-phase gate (logical subspace)
phases      = range(0, π, length=21)[2:end]   # θ = 0 is the identity: T* = 0 without any pulse
n_starts    = 10        # random starts per phase, plus the solution of the previous phase
time_weight = 5e-3      # first stage: J = φ + time_weight*T pushes each start towards a short T
loose_tol   = 1e-2      # first-stage results above this infidelity are not polished
gate_tol    = 1e-6      # a polished, converged result counts as a gate below this infidelity
tmin0       = 0.5       # initial T window (tstar_min, tstar_max), widened when necessary
tmax0       = 10.0
tmax_limit  = 40.0

# all gates (converged, infidelity < gate_tol) found for one phase in the window (tmin, tmax)
function find_gates(phase, extra_starts, tmin, tmax)
    tar = build_tar(gate_type, phase, n_qubits, algebra)
    solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tmin, tmax)
    nsteps = round(Int, tmax/dt)                 # dt = T/nsteps ≤ 1e-2 for every T in the window
    starts = [(randn(d), tmin + rand()*(tmax - tmin)) for _ in 1:n_starts]
    append!(starts, extra_starts)
    gates = []
    for (m0, T0) in starts
        T0 = clamp(T0, tmin + 0.01*(tmax - tmin), tmax - 0.01*(tmax - tmin))   # strictly inside the window
        # stage 1: with time penalty
        res1 = run_bfgs(m0, T0, algebra, solver, tar; time_weight=time_weight, nsteps=nsteps)
        m1, T1 = minimizer_mT(res1, solver)
        Optim.minimum(res1) - time_weight*T1 > loose_tol && continue
        # stage 2: polish without time penalty, so T settles at the gate time itself
        res2 = run_bfgs(m1, T1, algebra, solver, tar; time_weight=0.0, nsteps=nsteps, iterations=300)
        m2, T2 = minimizer_mT(res2, solver)
        cost = new_propagate(m2, algebra, solver, tar; T=T2, nsteps=nsteps, track_min=false).min_cost
        if Optim.converged(res2) && cost < gate_tol
            push!(gates, (T = T2, cost = cost, m = m2))
        end
    end
    return gates
end

function sweep(phases)
    results = []
    prev = nothing
    for phase in phases
        tmin, tmax = tmin0, tmax0
        extra = prev === nothing ? [] : [(prev.m, prev.T)]       # continuation from the previous phase
        while true
            gates = find_gates(phase, extra, tmin, tmax)
            if isempty(gates)
                if tmax >= tmax_limit
                    @warn "no gate found for θ = $(phase/π) π up to T = $tmax_limit"
                    break
                end
                tmax = min(1.5*tmax, tmax_limit)                 # widen the window upwards
                continue
            end
            best = gates[argmin([g.T for g in gates])]           # minimal-time gate
            if best.T < tmin + 0.02*(tmax - tmin) && tmin > 0.05
                tmin /= 2                                        # pressed against the lower bound: widen downwards
                continue
            end
            @printf("θ = %.3f π: T* = %.4f  infidelity = %.2e  (%d gates found, window (%.2f, %.1f))\n",
                    phase/π, best.T, best.cost, length(gates), tmin, tmax)
            push!(results, (phase = phase, T = best.T, cost = best.cost, m = best.m, n_gates = length(gates)))
            prev = best
            break
        end
    end
    return results
end

Random.seed!(1)
@time results = sweep(phases)

# ── plot ──────────────────────────────────────────────────────────────────────
θ    = [r.phase/π for r in results]
Tstar = [r.T for r in results]
logc = log10.([r.cost for r in results])

p = plot(θ, Tstar; color = :gray, alpha = 0.4, label = false,
         xlabel = L"\theta / \pi", ylabel = L"T^*",
         title = "minimal gate time vs. target phase", xlims = (0, 1.02), ylims = (0, 1.1*maximum(Tstar)))
scatter!(p, θ, Tstar; zcolor = logc, color = cgrad(["#cfe0f5", "#2a78d6", "#0b2f5e"]),
         markersize = 7, markerstrokewidth = 0.5, label = false,
         colorbar_title = L"\log_{10}(\mathrm{infidelity})")

savefig(p, joinpath(@__DIR__, "..", "res", "T_vs_phase.png"))
jldsave(joinpath(@__DIR__, "..", "res", "T_vs_phase.jld2");
        phases = [r.phase for r in results], T = Tstar, cost = [r.cost for r in results],
        m = reduce(hcat, [r.m for r in results]), n_gates = [r.n_gates for r in results])
p
