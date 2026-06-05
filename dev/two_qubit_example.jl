include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, ProgressBars, Plots

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
tmax = 11
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)

function obtain_optimal_time_dist(phase, algebra, solver; m0=nothing, method=Midpoint())
    stor = Storage{ComplexF64}(size(algebra.lie_basis[1],1), length(algebra.lie_basis))
    ryd_target = gate_k1(2, phase)
    target = Matrix{ComplexF64}(undef, 4, 4)
    project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    tar = TargetContainer(target)
    # println("Warm start with dt=1e-1")
    # m_warm_start = find_best_initial_costate(tar, algebra, solver, stor, method, m0=m0, dt=1e-1, show_trace=true, show_every=100)
    # println("Main optimisation with dt=1e-2")
    m_best = find_best_initial_costate(tar, algebra, solver, stor, method, m0=m0, dt=5e-2, show_trace=true, show_every=100, iterations=1500)
    sol, tracker = propagate(m_best, algebra, solver, stor, tar, method, dt=1e-3, full_results=true)
    return m_best, tracker.min_dist, tracker.tstar
end 
println("Setup finished")

##
phase_number = 10
phase_center = π
half = phase_number ÷ 2

# indices: center outward left and right
phases_right = [mod(phase_center + k * 2π/phase_number, 2π) for k in 0:half]
phases_left  = [mod(phase_center - k * 2π/phase_number, 2π) for k in 1:half]

m_lst_right    = [zeros(Float64, length(algebra.p_basis)) for _ in 1:length(phases_right)]
dist_lst_right = zeros(Float64, length(phases_right))
t_lst_right    = zeros(Float64, length(phases_right))

m_lst_left     = [zeros(Float64, length(algebra.p_basis)) for _ in 1:length(phases_left)]
dist_lst_left  = zeros(Float64, length(phases_left))
t_lst_left     = zeros(Float64, length(phases_left))

# right sweep: π → 2π
for (k, phase) in enumerate(ProgressBar(phases_right))
    m0 = k == 1 ? nothing : m_lst_right[k-1]
    println("Phase/2π: $(phase/(2π))")
    m_lst_right[k], dist_lst_right[k], t_lst_right[k] = obtain_optimal_time_dist(phase, algebra, solver, m0)
    println("dmin: $(dist_lst_right[k])")
    println("---")
    println("-")
end

# left sweep: π → 0, warm start from π result
for (k, phase) in enumerate(ProgressBar(phases_left))
    m0 = k == 1 ? m_lst_right[1] : m_lst_left[k-1]
    println("Phase/2π: $(phase/(2π))")
    m_lst_left[k], dist_lst_left[k], t_lst_left[k] = obtain_optimal_time_dist(phase, algebra, solver, m0)
    println("dmin: $(dist_lst_left[k])")
    println("---")
    println("-")
end

# combine
phase_lst = [reverse(phases_left); phases_right]
dist_lst  = [reverse(dist_lst_left); dist_lst_right]
t_lst     = [reverse(t_lst_left); t_lst_right]
m_lst = [reverse(m_lst_left); m_lst_right]

sort_idx  = sortperm(phase_lst)
phase_lst, dist_lst, t_lst, m_lst = phase_lst[sort_idx], dist_lst[sort_idx], t_lst[sort_idx], m_lst[sort_idx]

# uni-directional collection
# phase_number = 10
# phase_lst = zeros(Float64, phase_number)
# m_lst = [zeros(Float64, length(algebra.p_basis)) for _ in 1:phase_number]
# dist_lst = zeros(Float64, phase_number)
# t_lst = zeros(Float64, phase_number)

# for i in ProgressBar(1:phase_number)
#     if i == 1
#         m0 = nothing
#         phase = pi
#     else 
#         m0 = m_lst[i-1]
#         phase = mod(phase - 2*pi/phase_number, 2*pi)
#     end 
#     println("phase = $(phase/(2π)):")
#     phase_lst[i] = phase
#     m_lst[i], dist_lst[i], t_lst[i] = obtain_optimal_time_dist(phase, algebra, solver, m0)
#     println("dmin: $(dist_lst[i])")
#     println("---")
# end 


##

p = scatter(phase_lst ./(2π), t_lst,
    zcolor            = log10.(dist_lst),
    marker            = :circle,
    markersize        = 4,
    markerstrokewidth = 0,
    colorbar_title    = "\nlog(dmin)",  # \n pushes title away from ticks
    colorbar_titlefontsize = 8,
    colorbar_tickfontsize  = 7,
    right_margin      = 5Plots.mm,
    xlabel            = "ϕ/2π",
    ylabel            = "t*",
    label             = false,
    c                 = :viridis,
    gridalpha         = 0.3,
    gridlinewidth     = 0.5,
    minorgrid         = true,
    minorgridalpha    = 0.15,
    minorgridlinewidth = 0.5,
    #xticks            = 0:0.1:1,
    #yticks            = range(minimum(t_lst), maximum(t_lst), length=10))
)

##
@time begin
    m_best = find_best_initial_costate(tar, algebra, solver, stor, dt=1e-1)
end