include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, ProgressBars, Plots, ProgressMeter

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

algebra = Algebra(im_control, im_drift)
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse

tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-10
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)

ryd_target = gate_k1(2, pi)
target = Matrix{ComplexF64}(undef, 4, 4)
project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
tar = TargetContainer(target)
println("Setup finished")
##

@time begin 
    m_warm_start = find_best_initial_costate(tar, algebra, solver, stor, Midpoint(), dt=1e-1, show_trace=true)
end 