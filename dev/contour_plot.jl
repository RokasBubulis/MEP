include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressMeter, JLD2

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# gate
k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse

# params
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
function obtain_optimal_time_dist(phase, algebra, solver; m0=nothing, dt_warmstart=1e-1, dt_main=1e-2, show_trace=false)
    stor = Storage(2, length(algebra.lie_basis))
    ryd_target = gate_k1(2, phase)
    target = Matrix{ComplexF64}(undef, 4, 4)
    project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    tar = TargetContainer(target)
    m_warm_start = find_best_initial_costate(tar, algebra, solver, stor, m0=m0, dt=dt_warmstart, show_trace=show_trace, show_every=100)
    m_best = find_best_initial_costate(tar, algebra, solver, stor, m0=m_warm_start, dt=dt_main, show_trace=show_trace, show_every=100)
    min_dist, tstar, _ = propagate(m_best, algebra, solver, stor, tar, save_ulist=true, dt=dt_main)
    return min_dist, tstar
end 
println("Setup finished")
##
dt = 1e-2
optimise_for_phi = pi
stor = Storage(2, length(algebra.lie_basis))
ryd_target = gate_k1(2, optimise_for_phi)
target = Matrix{ComplexF64}(undef, 4, 4)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
m_best = find_best_initial_costate(tar, algebra, solver, stor, dt=dt)

##
max_dev = 0.05
phases = range((1-max_dev)*optimise_for_phi, (1+max_dev)*optimise_for_phi, 300)
results = []
for phase in phases 

    stor = Storage(2, length(algebra.lie_basis))
    ryd_target = gate_k1(2, phase)
    target = Matrix{ComplexF64}(undef, 4, 4)
    project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    tar = TargetContainer(target)
    d, tstar, Ulist = propagate(m_best, algebra, solver, stor, tar, save_ulist=true, dt=dt)
    n = length(Ulist)
    ts = dt .* (1:n)
    dists = [distance(Ulist[i], tar, algebra, stor) for i in eachindex(ts)]
    push!(results, (phase=phase, d=d, tstar=tstar, Ulist=Ulist, dists=dists, ts=ts))
end 

##
t_min_idx = 675
t_max_idx = 710
phases_arr = [r.phase   for r in results]
d_arr      = [r.d       for r in results]
ts_arr     = [r.ts[t_min_idx:t_max_idx]      for r in results]
dists_arr  = [r.dists[t_min_idx:t_max_idx]   for r in results]

D = hcat([log10.(v) for v in dists_arr]...)
p = contourf(phases./(pi), ts_arr[1], D;
    colorbar=true,
    c=:viridis,
    xlabel="ϕ/π",
    ylabel="t",
    colorbar_title="log(d(t))",
    size=(800, 500),
    levels=50,
    title="gate: CPhase(ϕ), m0 optimised for ϕ/π=$(optimise_for_phi/pi)"

    )
display(p)