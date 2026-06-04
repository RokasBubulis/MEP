include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, ProgressBars, Plots, ProgressMeter

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse
println("Setup finished")
##

tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-10
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)


function obtain_optimal_time_dist(phase, algebra, solver; method=Midpoint())
    stor = Storage{ComplexF64}(size(algebra.lie_basis[1],1), length(algebra.lie_basis))
    ryd_target = gate_k1(2, phase)
    target = Matrix{ComplexF64}(undef, 4, 4)
    project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    tar = TargetContainer(target)

    m_warm_start = find_best_initial_costate(tar, algebra, solver, stor, method, dt=1e-1, show_trace=false,)
    m_best = find_best_initial_costate(tar, algebra, solver, stor, method, m0=m_warm_start, dt=1e-2, show_trace=false)
    sol, tracker = propagate(m_warm_start, algebra, solver, stor, tar, method, dt=1e-2, full_results=true)
    return tracker
end 
##
phase_number = 20
phases    = range(0, 2*pi, length=phase_number) 
tstars    = Vector{Float64}(undef, phase_number)
min_dists = Vector{Float64}(undef, phase_number)
@assert Threads.nthreads() != 1
p = Progress(phase_number)
Threads.@threads for i in 1:phase_number
    phase = phases[i]
    tracker = obtain_optimal_time_dist(phase, algebra, solver)
    tstars[i]    = tracker.tstar
    min_dists[i] = tracker.min_dist
    next!(p)
end
##
p1 = scatter(collect(phases), tstars,
    zcolor         = log10.(min_dists),
    marker         = :circle,
    markersize     = 4,
    markerstrokewidth = 0,
    colorbar_title = "mlog₁₀dmin",
    xlabel         = "ϕ",
    ylabel         = "t*",
    label          = false,
    c              = :viridis)

