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
phase_number = 10
phase_lst = range(0, pi, phase_number)
m_number = 5
t_lst = zeros(Float64, phase_number, m_number)
dist_lst = zeros(Float64, phase_number, m_number)

println("Starting parallel computation: $phase_number phases × $m_number initialisations across $(Threads.nthreads()) threads")
flush(stdout)
N = phase_number * m_number
p = Progress(N; desc="", barlen=40)
Threads.@threads for idx in 1:N
    i = div(idx-1, m_number) + 1
    j = mod(idx-1, m_number) + 1
    phase = phase_lst[i]

    # println("  [Thread $tid] Starting (i=$i, j=$j) phase = $(round(phase, digits=4))")
    # flush(stdout)

    local_m0 = rand(Float64, length(algebra.p_basis))
    local_m0 ./= norm(local_m0)
    dist_lst[i,j], t_lst[i,j] = obtain_optimal_time_dist(phase, algebra, solver, m0=local_m0)

    # println("  [Thread $tid] Done (i=$i, j=$j) → dist = $(round(dist_lst[i,j], digits=6)), t = $(round(t_lst[i,j], digits=6))")
    # flush(stdout)
    next!(p)
end

println("\nAll phases complete.")
println("  dist_lst =\n", round.(dist_lst, digits=6))
println("  t_lst =\n",    round.(t_lst,    digits=6))

println("\nAll phases complete.")
println("  dist_lst = ", round.(dist_lst, digits=6))
println("  t_lst    = ", round.(t_lst,    digits=6))
##
p = make_plot_multi_m(phase_lst, t_lst, dist_lst)
title!(p, "$m_number random initial m0")
##
# m0 = rand(Float64, length(algebra.p_basis))
# m0 /= norm(m0)
m_best = [0.27212694808902316, 0.3559272634222538, -0.02590904666581743, 0.2887486241577902, -0.2323137997060933, 0.04824964971227963, 0.8117375994698723]
m, d, t = obtain_optimal_time_dist(0.8*pi, algebra, solver, m0=m_best, show_trace=true, dt_main=1e-2)

##
stor = Storage(2, length(algebra.lie_basis))
ryd_target = gate_k1(2, 0.2*pi)
target = Matrix{ComplexF64}(undef, 4, 4)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
mstart = find_best_initial_costate_cma(tar, algebra, solver, stor, dt=1e-1, sigma=1)
mbest = find_best_initial_costate_cma(tar, algebra, solver, stor, m0=mstart, dt=1e-1, sigma=0.3)