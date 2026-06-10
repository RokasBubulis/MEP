include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, Plots

# generators
n_qubits = 2
tmax = 30
im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
dim = size(im_control, 1)

algebra = Algebra(im_control, im_drift)
stor = Storage(n_qubits, length(algebra.lie_basis))

k0 = ComplexF64[1.0; 0.0; 0.0]
k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k0(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k0*k0' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse

reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
dt = 1e-2

ryd_target = gate_k1(n_qubits, pi)
target = Matrix{ComplexF64}(undef, 2^n_qubits, 2^n_qubits)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
tar = TargetContainer(ryd_target)
println("Setup finished")
##
results = []

for j in 1:5
    m0 = rand(Float64, length(algebra.p_basis))
    m0 /= norm(m0)
    min_dist, tstar, Ulist = propagate(m0, algebra, solver, stor, tar, dt=dt, save_ulist=true)
    dists = [distance(U, tar, algebra, stor) for U in Ulist]
    ts = dt .* (1:length(Ulist))
    push!(results, (m0=m0, ts=ts, dists=dists, min_dist=min_dist, tstar=tstar))
end

p = plot(grid=true, minorgrid=true, gridalpha=0.5, minorgridalpha=0.2)
for (j, r) in enumerate(results)
    plot!(p, r.ts[1:end-1], r.dists[1:end-1], label="$j: dmin=$(round(r.min_dist, sigdigits=2)), t*=$(round(r.tstar, sigdigits=2))")
end
xlabel!(p, "t")
ylabel!(p, "dmin")
title!(p, "n = $n_qubits, target levels = $(round(Int, (size(tar.target,1)^(1/n_qubits))))")#, legend=:topright)
display(p)

##
m0 = rand(Float64, length(algebra.p_basis))
m0 /= norm(m0)
m_best = find_best_initial_costate_finite_diff(tar, algebra, solver, stor, dt=dt, show_trace=true, m0=m0, show_every=1)
min_dist, tstar, Ulist = propagate(m_best, algebra, solver, stor, tar, dt=dt, save_ulist=true)
println("dmin=$min_dist, t*=$tstar")

##

m_best = find_best_initial_costate(tar, algebra, solver, stor, dt=dt, show_trace=true, m0=m0, show_every=1)
min_dist, tstar, Ulist = propagate(m_best, algebra, solver, stor, tar, dt=dt, save_ulist=true)
println("dmin=$min_dist, t*=$tstar")

##
ryd_target = gate_k1(n_qubits, 0.1*pi)
target = Matrix{ComplexF64}(undef, 2^n_qubits, 2^n_qubits)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
m_best = find_best_initial_costate(tar, algebra, solver, stor, dt=dt, show_trace=true, show_every=1)
min_dist, tstar, _ = propagate(m_best, algebra, solver, stor, tar, dt=dt, save_ulist=true)
println("mbest: $(round.(m_best, sigdigits=3)), dmin=$min_dist, t*=$tstar")