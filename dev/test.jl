include("../src/structs.jl")
include("../src/propagation.jl")
using Plots 
using ProgressBars
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
dt = 1e0
adaptive_for_RK=false
solver = SolverParams(tmax, reltol, abstol, dist_tol, opt_tol)
m0 = copy(algebra.neg_im_drift_lie)[2:end]
println("Running Ut")
Ut = propagate(m0, algebra, system, solver, stor, Vern9(), dt=1e-4, adaptive=true, saveat=solver.tmax, return_sol=true)[end].U
# println(Ut)

method_num = 4
dt_size = 20
errs = zeros(Float64, (method_num, dt_size))
dt_list = logrange(1e-3, 1e0, dt_size)
results = Dict()
methods = [Midpoint(), RK4(), Tsit5(), OwrenZen5()]
for (i,method) in enumerate(methods)
    println("Method $method")
    for j in ProgressBar(1:dt_size)
        dt = dt_list[j]
        #m_best, dmin = find_best_initial_costate(algebra, system, solver, stor, method, dt=dt, adaptive=adaptive)
        sol = propagate(m0, algebra, system, solver, stor, method, dt=dt, saveat=solver.tmax, return_sol=true, adaptive=false)[end].U
        errs[i,j] = max(1 - 1/9*abs(tr(adjoint(sol) * Ut)), 1e-14)
    end
    # plot!(p, sol.t, dists, yscale=:log10, grid=true, gridlinewidth=0.5, gridalpha=0.4, 
    # minorgrid=true, minorgridalpha=0.2, label=
    # "$method_name : dmin = $(round((dmin),sigdigits=2)) at t = $(round(results[method_name].min_time,sigdigits=3))")
end
##


p = plot()

for i in eachindex(methods)
    plot!(p, dt_list, max.(errs[i,:], [1e-16]), xscale=:log10, yscale=:log10, label="Method: $i")
end 
xlabel!(p, "t")
ylabel!(p, "dmin to target coset")
title!(p, "dt=$dt, adaptive for RK4=$adaptive_for_RK", legend=:bottomleft)
if adaptive_for_RK
    name = "results/dmin_method_comparison_$adaptive_for_RK.png"
else 
    name = "results/dmin_method_comparison_dt_$dt.png"
end 
#savefig(p, name)
display(p)