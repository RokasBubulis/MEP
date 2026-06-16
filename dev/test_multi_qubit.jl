include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, Plots

# generators
n_qubits = 2
tmax = 10
im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
tstar_min = 3
tstar_max = 5
phase = pi / 5
gate_type = "k1"
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
tar = build_tar(gate_type, phase, n_qubits, algebra)
println("Setup finished")

##
dt=1e-2
m_warm, dist = find_best_initial_costate_bbo(tar, algebra, solver, dt=dt)
##
##
dt=1e-2
m_best = find_best_initial_costate(tar, algebra, solver, m0=m_warm, dt=dt, show_trace=true)
min_dist, tstar, _, _ = propagate(m_best, algebra, solver, tar, dt=dt, save=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")

##
dt=1e-2
m_best = find_best_initial_costate(tar, algebra, solver, m0=m_warm, dt=dt, show_trace=true)
min_dist, tstar, _, _ = propagate(m_best, algebra, solver, tar, dt=dt, save=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")



##
println("CMA Warm start")
m0 = rand(Float64, length(algebra.p_basis))
m0 /= norm(m0)
m_warm_cma= find_best_initial_costate_cma(tar, algebra, solver, stor, dt=dt, m0=m0, show_trace=true, sigma=1.0, iterations=1000)
min_dist, tstar, _, _ = propagate(m_warm_cma, algebra, solver, stor, tar, dt=dt, save=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")

##
println("NM Main using CMA warm start")
dt=1e-2
m_best_nm, m_log_main_nm = find_best_initial_costate_with_logging(tar, algebra, solver, stor, dt=dt, m0=m_warm_cma, show_trace=true, show_every=100)
min_dist, tstar, _, _ = propagate(m_best_nm, algebra, solver, stor, tar, dt=dt, save=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")

##
# results = []

# for j in 1:5
#     m0 = rand(Float64, length(algebra.p_basis))
#     m0 /= norm(m0)
#     min_dist, tstar, Ulist = propagate(m0, algebra, solver, stor, tar, dt=dt, save_ulist=true)
#     dists = [distance(U, tar, algebra, stor) for U in Ulist]
#     ts = dt .* (1:length(Ulist))
#     push!(results, (m0=m0, ts=ts, dists=dists, min_dist=min_dist, tstar=tstar))
# end

# p = plot(grid=true, minorgrid=true, gridalpha=0.5, minorgridalpha=0.2)
# for (j, r) in enumerate(results)
#     plot!(p, r.ts[1:end-1], r.dists[1:end-1], label="$j: dmin=$(round(r.min_dist, sigdigits=2)), t*=$(round(r.tstar, sigdigits=2))")
# end
# xlabel!(p, "t")
# ylabel!(p, "dmin")
# title!(p, "n = $n_qubits, target levels = $(round(Int, (size(tar.target,1)^(1/n_qubits))))")#, legend=:topright)
# display(p)

##
# println("NM Warm start")
# dt=1e-1
# m_warm = find_best_initial_costate(tar, algebra, solver, stor, dt=dt, show_trace=true, show_every=100, iterations=3500)
# min_dist, tstar, Ulist = propagate(m_warm, algebra, solver, stor, tar, dt=dt, save_ulist=true)
# println("dt: $dt, dmin=$min_dist, t*=$tstar")


# 3q, pi/2, k1: 
tar = build_tar("k1", pi/2, n_qubits, algebra)
dt=1e-1
m0 = [0.5220387770264742
 0.45274842212811955
 0.002595608672899914
 0.011925525913954241
 0.1582266062981138
 0.24890051522446854
 0.04523696843895026
 0.5386378384051406
 0.23180717562307362
 0.2990762943183803]

m_warm_cma= find_best_initial_costate_cma(tar, algebra, solver, stor, dt=dt, m0=m0, show_trace=true, sigma=1.0, iterations=1000)
min_dist, tstar, _, _ = propagate(m_warm_cma, algebra, solver, stor, tar, dt=dt, save=true)
println("dt: $dt, dmin=$min_dist, t*=$tstar")