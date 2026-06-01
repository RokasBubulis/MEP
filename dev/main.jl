include("../src/structs.jl")
include("../src/propagation.jl")
using Plots 

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

tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
opt_tol = 1e-8
dt = 1e-3
adaptive=false
solver = SolverParams(tmax, reltol, abstol, dist_tol, opt_tol)

results = Dict()

p = plot()
for method in [Midpoint(), RK4()]
    m_best = find_best_initial_costate(algebra, system, solver, stor, method, dt=dt, adaptive=adaptive)
    sol = propagate(m_best, algebra, system, solver, stor, method, dt=dt, saveat=[], return_sol=true)
    Us = [u.U for u in sol.u]
    dists = [distance(Us[i], system, solver, stor) for i in eachindex(sol.t)]
    min_dist = minimum(dists)
    min_time = sol.t[argmin(dists)]
    method_name = nameof(typeof(method))
    results[method_name] = (m_best=m_best, sol=sol, Us=Us, dists=dists, min_dist = min_dist, min_time=min_time)

    plot!(p, sol.t, dists, yscale=:log10, grid=true, gridlinewidth=0.5, gridalpha=0.4, 
    minorgrid=true, minorgridalpha=0.2, label=
    "$method_name : dmin = $(round((results[method_name].min_dist),sigdigits=2)) at t = $(round(results[method_name].min_time,sigdigits=3))")
end

xlabel!(p, "t")
ylabel!(p, "dmin to target coset")
title!(p, "dt=$dt, adaptive=$adaptive", legend=:bottomleft)
if adaptive
    name = "results/dmin_method_comparison_$adaptive.png"
else 
    name = "results/dmin_method_comparison_dt_$dt.png"
end 
savefig(p, name)
display(p)