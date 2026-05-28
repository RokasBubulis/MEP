include("structs.jl")
include("optimisation.jl")
include("../liepmp/src/unwrap_evolution.jl")

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

##

m0 = rand(length(algebra.p_basis))
# solver parameters
tmax = 10
tol = 1e-10

dt_rk = 1e-4
solver = SolverParams(tmax, dt_rk, tol)
stor_rk = Storage{ComplexF64}(dim, length(algebra.lie_basis))
Us_rk = propagate_RK4(m0, algebra, system, solver, stor_rk, save_only_U=true)[end]
# unwrapped_phases_rk, _ = unwrap_eigenvalues_over_time(Us_rk)
# final_phases_rk = unwrapped_phases_rk[end]

# Error propagation analysis
dts = logrange(1e-2, 0.5, 12)
max_errors = Vector{Float64}(undef, length(dts))
sum_errors = Vector{Float64}(undef, length(dts))
println("Running with $(Threads.nthreads()) threads")
for i in eachindex(dts)
    #println("i=$i on thread $(Threads.threadid())")
    dt = dts[i]
    solver = SolverParams(tmax, dt, tol)

    stor_mp = Storage{ComplexF64}(dim, length(algebra.lie_basis))
    Us_mp = propagate_MP(m0, algebra, system, solver, stor_mp, save_only_U=true)[end]

    errors = 1 - 1/9*abs(tr(adjoint(Us_rk) * Us_mp))
    max_errors[i] = errors
    # sum_errors[i] = sum(errors)
end

p = plot()
y = [max_errors[i] > 0.0 ? max_errors[i] : NaN for i in eachindex(dts)]
#s = [sum_errors[i] > 0.0 ? sum_errors[i] : NaN for i in eachindex(dts)]
plot!(p, dts, y, label="Max abs error", xscale=:log10, yscale=:log10, marker=:circle, markersize=3, markercolor=:auto, grid=true)
#plot!(p, dts, s, label="Sum abs error", yscale=:log10, marker=:circle, markersize=3, markercolor=:auto, grid=true)

xlabel!(p, "dt")
ylabel!(p, "|ϵ(T,dt)_MP - ϵ(T,$dt_rk)_RK4|")
title!("T = $tmax")
display(p)