include("../src/structs.jl")
include("../src/propagation.jl")
using Plots, ProgressBars

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

m0 = copy(algebra.neg_im_drift_lie)[2:end]
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
opt_tol = 1e-8
adaptive=false
solver = SolverParams(tmax, reltol, abstol, dist_tol, opt_tol)
n = size(system.im_control, 1)

method_baseline = Vern9()
dt_baseline = 1e-4
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))
sol = propagate(m0, algebra, system, solver, stor, method_baseline, dt=dt_baseline, adaptive=true, saveat=solver.tmax, return_sol=true)
U_baseline = copy(stor.U)
println("Baseline computed")
##
dt_number = 10
methods_lst = [Midpoint(), RK4(), Tsit5(), Vern6()]
dts = logrange(1e-4, 5e-1, dt_number)

errs = zeros(Float64, (length(methods_lst), dt_number))
Uts = Matrix{Matrix{ComplexF64}}(undef, length(methods_lst), dt_number)
for (i,method) in enumerate(methods_lst)
    println("Method: $(nameof(typeof(method)))")
    for j in ProgressBar(1:dt_number)
        dt = dts[j]
        stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))
        sol = propagate(m0, algebra, system, solver, stor, method, dt=dt, adaptive=false, saveat=solver.tmax, return_sol=true)
        Uts[i,j] = copy(stor.U)
        errs[i,j] = 1 - 1/n * abs(tr(adjoint(U_baseline) * Uts[i,j]))
    end 
end 

## 
p = plot()
error_lst = []
for (i,method) in enumerate(methods_lst)
    
    devs = [norm(Uts[i,j]' * Uts[i,j] - I) for j in 1:dt_number]
    plot!(dts, errs[i,:], label="$(nameof(typeof(method))), max unitarity error: $(round(maximum(devs),sigdigits=4))", xscale=:log10, yscale=:log10, grid=true, gridlinewidth=0.5, gridalpha=0.4, 
    minorgrid=true, minorgridalpha=0.2,marker=:circle)
end 

xlabel!(p, "dt")
ylabel!(p, "U(t=$(solver.tmax)) overlap error with baseline")
title!(p, "Baseline: $(nameof(typeof(method_baseline))) at dt=$dt_baseline", legend=:topleft)
display(p)
