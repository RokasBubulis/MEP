include("../src/structs.jl")
include("../src/propagation.jl")
using Plots 
using ForwardDiff
# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# prepare system struct 
target = Matrix(SparseMatrixCSC{ComplexF64, Int}(I, dim, dim))
target[5,5] = -1.0
title="CZ"
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
system = System{ComplexF64}(im_control, im_drift, target, target_logic)
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))
println("Setup finished")
##

# a = Matrix{ComplexF64}(undef, 4, 4)
# b = Matrix{ComplexF64}(undef, 4, 4)

# U = randn(ComplexF64, 9, 9)
# project_propagator_to_logical_subspace!(a, U)
# project_from_3_to_2_levels!(b, U, system, stor)
# display(a-b)

tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
opt_tol = 1e-8
dt = 1e-2
adaptive=true
solver = SolverParams(tmax, reltol, abstol, dist_tol, opt_tol)

results = Dict()
m0 = rand(length(algebra.p_basis))
p = plot()
for method in [Vern7()]

    m_best, dmin = find_best_initial_costate(algebra, system, solver, stor, method, dt=dt, adaptive=false)
    sol = propagate(m_best, algebra, system, solver, stor, method, dt=dt, saveat=[], return_sol=true)
    Us = [u.U for u in sol.u]
    dists = [distance(Us[i], system, solver, stor) for i in eachindex(sol.t)]
    min_dist = minimum(dists)
    min_time = sol.t[argmin(dists)]
    method_name = nameof(typeof(method))
    results[method_name] = (m_best=m0, dmin=dmin, sol=sol, Us=Us, dists=dists, min_dist = min_dist, min_time=min_time)

    plot!(p, sol.t, dists, yscale=:log10, grid=true, gridlinewidth=0.5, gridalpha=0.4, 
    minorgrid=true, minorgridalpha=0.2, label=
    "$method_name : dmin = $(round((dmin),sigdigits=2)) at t = $(round(results[method_name].min_time,sigdigits=3))")
end

xlabel!(p, "t")
ylabel!(p, "dmin to target coset")
title!(p, "dt=$dt", legend=:bottomleft)
# if adaptive_for_RK
#     name = "results/dmin_method_comparison_$adaptive_for_RK.png"
# else 
#     name = "results/dmin_method_comparison_dt_$dt.png"
# end 
#savefig(p, name)
display(p)