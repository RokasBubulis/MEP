# Check that the saved optimal control profiles α_opt(t) produce the intended gates.
# For every phase Φ in res/HJB_T_vs_phase.jld2 and res/PXP_T_vs_phase.jld2, propagate from U(0) = I
#     dU/dt = H_opt(α(t)) U,    H_opt(α) = e^{-αC} (-D) e^{αC}     (C = i·control, D = i·drift of the system's generators)
# using only the α(t) samples (no costate, no Lie-algebra coordinates, no Krylov), in two ways:
#   A) midpoint-averaged steps as in new_propagate:  U_{k+1} = exp(dt (H_opt(α_k) + H_opt(α_{k+1}))/2) U_k
#   B) continuous time: α(t) = natural cubic spline through the samples, adaptive Vern9, tolerance 1e-12
# and compare U(T*) with C-Φ = diag(1, 1, 1, e^{iΦ}) up to single-qubit phases θ and a global phase.
# Negative control: the HJB profile propagated with the PXP generators and vice versa.
# Output: printed table and res/alpha_verification_HJB_PXP.txt
# Run:  julia --project=. dev/verify_alpha_HJB_PXP.jl

include("../src/structs.jl")      # generators (construct_HJB_generators, construct_PXP_generators)
using JLD2, LinearAlgebra, DifferentialEquations, Printf

systems = Dict("HJB" => construct_HJB_generators, "PXP" => construct_PXP_generators)

# ── generator of the evolution for a given α ──────────────────────────────────
struct System
    c::Vector{ComplexF64}           # diagonal of C = i·control (the control is diagonal)
    mD::Matrix{ComplexF64}          # -D = -i·drift
end
function System(gen)
    control, drift = gen(2)
    @assert isdiag(Matrix(control)) "control generator is expected to be diagonal"
    System(im .* diag(Matrix(control)), -im .* Matrix(drift))
end
# e^{-αC} (-D) e^{αC} with C diagonal: entry (j,k) is multiplied by e^{-α c_j} e^{α c_k}
H_opt(s::System, α) = (e = exp.(α .* s.c); conj.(e) .* s.mD .* transpose(e))   # c is imaginary ⇒ e^{-αc} = conj(e^{αc})

# ── natural cubic spline through (t_k, α_k) ──────────────────────────────────
function cubic_spline(t, y)
    n = length(t); h = diff(t)
    dl, d, du, r = zeros(n-1), ones(n), zeros(n-1), zeros(n)
    for i in 2:n-1
        dl[i-1], d[i], du[i] = h[i-1], 2(h[i-1] + h[i]), h[i]
        r[i] = 6((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    end
    M = Tridiagonal(dl, d, du) \ r
    return function (x)
        i = clamp(searchsortedlast(t, x), 1, n - 1)
        a, b = t[i+1] - x, x - t[i]
        return (M[i]*a^3 + M[i+1]*b^3)/(6h[i]) + (y[i]/h[i] - M[i]*h[i]/6)*a + (y[i+1]/h[i] - M[i+1]*h[i]/6)*b
    end
end

function propagate_steps(s::System, ts, αs)
    U = Matrix{ComplexF64}(I, 9, 9)
    for k in 1:length(ts)-1
        U = exp((ts[k+1] - ts[k]) * (H_opt(s, αs[k]) + H_opt(s, αs[k+1]))/2) * U
    end
    return U
end

function propagate_continuous(s::System, ts, αs)
    α = cubic_spline(ts, αs)
    rhs!(dU, U, p, t) = mul!(dU, H_opt(s, α(t)), U)
    prob = ODEProblem(rhs!, Matrix{ComplexF64}(I, 9, 9), (ts[1], ts[end]))
    return solve(prob, Vern9(); abstol=1e-12, reltol=1e-12, save_everystep=false).u[end]
end

# ── gate quality ─────────────────────────────────────────────────────────────
logical = [1, 2, 4, 5]               # |00⟩, |01⟩, |10⟩, |11⟩ in the basis |q1 q2⟩, q ∈ {0, 1, r}
nones   = [0, 1, 1, 2]               # number of qubits in |1⟩
function gate_report(U, Φ)
    u = U[logical, logical]
    target = [1, 1, 1, exp(im*Φ)]
    f(θ) = abs(sum(conj(target .* exp.(im*θ .* nones)) .* diag(u)))/4     # |tr((Z(θ)⊗Z(θ) U_T)† u)|/4
    F = maximum(f, range(-π, π, length=200_001))
    leak = maximum(1 - sum(abs2, U[logical, x]) for x in logical)       # population left outside the logical states
    offd = norm(u - Diagonal(diag(u)))                                   # transitions between logical states
    Φa = mod(angle(u[1,1] * conj(u[2,2]) * conj(u[3,3]) * u[4,4]), 2π)   # controlled phase (θ- and global-phase-free)
    return (φ = 1 - F, leak = leak, offd = offd, Φa = Φa)
end

# ── run ──────────────────────────────────────────────────────────────────────
lines = String[]
function report(io_lines, sysname, propname, Φ, T, method, r)
    l = @sprintf("%-4s %-4s %.3f  %.4f  %-12s %.2e   %.2e   %.1e   %.6f   %+.1e",
                 sysname, propname, Φ/π, T, method, r.φ, r.leak, r.offd, r.Φa/π, (r.Φa - Φ)/π)
    println(l); push!(io_lines, l)
end
header = "sys  gen  Φ/π    T*      method       infidelity leakage    off-diag  Φ_a/π      (Φ_a-Φ)/π"
println(header)
data = Dict(name => load(joinpath(@__DIR__, "..", "res", "$(name)_T_vs_phase.jld2")) for name in keys(systems))
sys  = Dict(name => System(gen) for (name, gen) in systems)
for name in ("HJB", "PXP"), k in eachindex(data[name]["Φ"])
    d = data[name]
    Φ, T, ts, αs = d["Φ"][k], d["T"][k], d["t"][k], d["alpha"][k]
    d["gate"][k] || continue
    report(lines, name, name, Φ, T, "A steps",      gate_report(propagate_steps(sys[name], ts, αs), Φ))
    report(lines, name, name, Φ, T, "B continuous", gate_report(propagate_continuous(sys[name], ts, αs), Φ))
end
println("\nnegative control: profile of one system propagated with the generators of the other")
for (name, other) in (("HJB", "PXP"), ("PXP", "HJB")), Φ in (π, 0.5π)
    d = data[name]; k = findfirst(x -> abs(x - Φ) < 1e-9, d["Φ"])
    report(lines, name, other, Φ, d["T"][k], "B continuous", gate_report(propagate_continuous(sys[other], d["t"][k], d["alpha"][k]), Φ))
end

open(joinpath(@__DIR__, "..", "res", "alpha_verification_HJB_PXP.txt"), "w") do io
    println(io, "# α_opt(t) from res/{HJB,PXP}_T_vs_phase.jld2 propagated from U = I with dU/dt = e^{-αC}(-D)e^{αC} U")
    println(io, "# sys = system the profile was optimised for, gen = generators used to propagate it")
    println(io, "# A: midpoint-averaged steps (as new_propagate); B: cubic-spline α(t), Vern9, tol 1e-12")
    println(io, "# infidelity vs C-Φ = diag(1,1,1,e^{iΦ}) up to single-qubit phases; leakage = max population outside |00>,|01>,|10>,|11>")
    println(io, "# " * header)
    foreach(l -> println(io, l), lines)
end
println("\nsaved res/alpha_verification_HJB_PXP.txt")
