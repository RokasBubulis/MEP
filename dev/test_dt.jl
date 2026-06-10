include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressBars, JLD2

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
tmax = 12
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
stor = Storage(2, length(algebra.lie_basis))
target = Matrix{ComplexF64}(undef, 4, 4)
ryd_target = gate_k1(2, pi)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
println("Setup finished")
##

dt_number = 10
dts = logrange(1e-3, 1e0, dt_number)
d_lst = zeros(Float64, dt_number)
for i in ProgressBar(1:dt_number)
    mbest = find_best_initial_costate(tar, algebra, solver, stor, dt=dts[i], show_trace=true)
    d_lst[i] = propagate(mbest, algebra, solver, stor, tar, dt=dts[i])
end 
##
p = plot(dts, d_lst, xscale=:log10, yscale=:log10, grid=true, minor_grid=true, marker=:circle, gridlinewidth=0.5, gridalpha=0.4, 
    minorgrid=true, minorgridalpha=0.2, label="Midpoint")
xlabel!(p, "Δt")
ylabel!(p, "Minimal objective (dmin)")
title!(p, "Optimisation for 2 qubit CZ", legend=:bottomright)
display(p)