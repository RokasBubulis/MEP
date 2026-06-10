include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressMeter, JLD2, Base.Threads

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
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
println("Setup finished")
##

vary_indices = [3, 4, 5, 6, 7, 8]
test_values = collect(0.1:0.05:0.5)
dt = 1e-2

n_vals = length(test_values)
n_idx  = length(vary_indices)
results_matrix = fill(NaN, n_vals, n_idx)
drift_lie_coeff = 5
total = n_vals * n_idx
p = Progress(total; desc="Computing: ", showspeed=true)

tasks = [(i, j, vary_indices[i], test_values[j]) for i in 1:n_idx, j in 1:n_vals]

Threads.@threads for (i, j, idx, val) in vec(tasks)
    lie_coeffs = zeros(8)
    lie_coeffs[2] = drift_lie_coeff
    lie_coeffs[idx] = val

    target = sparse(exp(-Matrix(
        sum(lie_coeffs[k] * algebra.lie_basis[k] for k in eachindex(lie_coeffs))
    )))
    tar = TargetContainer(target)
    stor = Storage(2, length(algebra.lie_basis))
    m_best = find_best_initial_costate(tar, algebra, solver, stor; show_trace=false, dt=dt)
    d = propagate(m_best, algebra, solver, stor, tar; dt=dt)

    results_matrix[j, i] = d
    next!(p)
end

# Rebuild the Dict for compatibility with downstream code
results = Dict(vary_indices[i] => results_matrix[:, i] for i in 1:n_idx)

##
plt = plot(
    title  = "c_2 = $drift_lie_coeff, tmax = $(solver.tmax)",
    xlabel = "Value of c_μ",
    ylabel = "Min distance",
    legend=:bottomright,
    grid=true,
    minorgrid=true,
    gridlinewidth=0.5, gridalpha=0.4, 
)
for (i, idx) in enumerate(vary_indices)
    plot!(plt, test_values, replace(results[idx], NaN => missing);
          label="μ=$idx, depth=$(algebra.depths[idx])", marker=:circle, yscale=:log10)
end
display(plt)
##