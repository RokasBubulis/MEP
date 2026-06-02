include("../src/structs.jl")
include("../src/propagation.jl")
using Plots 
using Base.Threads
using ProgressMeter

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
grad_tol = 1e-8
unitary_tol = 1e-10
adaptive=false
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)

dt_number = 12
methods_lst = [Midpoint()]
dts = logrange(1e0, 5e-1, dt_number)

min_dists = zeros(Float64, length(methods_lst), dt_number)
@assert Threads.nthreads() != 1
for (i, method) in enumerate(methods_lst)
    println("Method: $(nameof(typeof(method)))")
    
    p = Progress(dt_number, desc="$(nameof(typeof(method))): ")
    
    @threads for j in 1:dt_number
        dt = dts[j]
        stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

        _, dmin = find_best_initial_costate(
            algebra, system, solver, stor, method;
            dt=dt, show_trace=false, adaptive=adaptive
        )

        min_dists[i, j] = dmin
        next!(p)  # thread-safe progress update
    end
end

## 
p = plot()
for i in eachindex(methods_lst)
    plot!(p, dts, min_dists[i, :], label="$(nameof(typeof(methods_lst[i])))", xscale=:log10, yscale=:log10, grid=true, gridlinewidth=0.5, gridalpha=0.4, 
    minorgrid=true, minorgridalpha=0.2,marker=:circle)
end 

xlabel!(p, "dt")
ylabel!(p, "dmin")
title!(p, "Minimum distance to target coset for CZ gate",legend=:bottomright)
display(p)

# dt_number = 10
# methods_lst = [Midpoint(), Tsit5()]
# dts = logrange(5e-3, 5e-1, dt_number)
# min_dists = zeros(Float64, (length(methods_lst), dt_number))
# for (i,method) in enumerate(methods_lst)
#     println("Method: $(nameof(typeof(method)))")
#     for j in ProgressBar(1:dt_number)
#         dt = dts[j]
#         stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))
#         m_best, dmin = find_best_initial_costate(algebra, system, solver, stor, method, dt=dt, show_trace=false, adaptive=adaptive)
#         min_dists[i,j] = dmin
#     end 
# end 