include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressBars, JLD2

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
im_drift .+= im * (operator(OopRyd([1]),2)*operator(RopRyd([2]),2) + operator(OopRyd([2]),2)*operator(RopRyd([1]),2))
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# gate
k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse

# params
tmax = 12
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
stor = Storage(2, length(algebra.lie_basis))
target = Matrix{ComplexF64}(undef, 4, 4)

function obtain_optimal_time_dist(phase, algebra, solver; m0=nothing, dt_warmstart=1e-1, dt_main=1e-2, show_trace=false)
    ryd_target = gate_k1(2, phase)
    project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    tar = TargetContainer(target)
    # println("Warm start with dt=1e-1")
    m_warm_start = find_best_initial_costate(tar, algebra, solver, stor, m0=m0, dt=dt_warmstart, show_trace=show_trace, show_every=100, iterations=1500)
    # println("Main optimisation with dt=1e-2")
    m_best = find_best_initial_costate(tar, algebra, solver, stor, m0=m_warm_start, dt=dt_main, show_trace=show_trace, show_every=100, iterations=1000)
    min_dist, tstar, _ = propagate(m_best, algebra, solver, stor, tar, save_ulist=true, dt=dt_main)
    return m_best, min_dist, tstar
end 
println("Setup finished")
##
m, d, t = obtain_optimal_time_dist(pi, algebra, solver, show_trace=true)