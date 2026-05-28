include("structs.jl")
include("optimisation.jl")

using Plots

drift_lie_coeff = 2.0

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# solver parameters
tmax = 3.0
dt = 1e-2
tol = 1e-8
lambda = 0.0
Newton_steps = 50
Newton_tol = 1e-10
Newton_damping = 1.0
solver = SolverParams(tmax, dt, tol, lambda, Newton_steps, Newton_tol, Newton_damping)

# orthogonalise drift wrt control
c1 = tr(im_control * adjoint(im_drift))
c2 = tr(im_control * adjoint(im_control))
im_drift_orthogonal = im_drift - c1/c2 * im_control
@assert isapprox(tr(im_drift_orthogonal * adjoint(im_control)), 0.0, atol=1e-10)

# normalise generators 
im_control /= norm(im_control)
im_drift_orthogonal /= norm(im_drift_orthogonal)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift_orthogonal)
# @assert length(lie_coeffs) == length(algebra.lie_basis)

# prepare mutable storage 
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

# Loop only over what changes
vary_indices = [3, 4, 6]
test_values = 0.1:0.1:1.0

results = Dict{Int, Vector{Float64}}()

for idx in vary_indices
    min_dists = Float64[]
    println("Computing for idx: $idx")
    for val in test_values
        # local lie_coeffs, target, system, m_best, dists
        lie_coeffs = zeros(8)
        lie_coeffs[2] = drift_lie_coeff
        lie_coeffs[idx] = val

        target = sparse(exp(-Matrix(
            sum(lie_coeffs[i] * algebra.lie_basis[i] for i in eachindex(lie_coeffs))
        )))
        system = System{ComplexF64}(im_control, im_drift_orthogonal, target)
        try 
            m_best = find_best_initial_costate_autograd(algebra, system, solver, stor; verbose = false)
            _, _, _, dists = propagate(m_best, algebra, system, solver, stor; save = true)
            push!(min_dists, abs(minimum(dists)))
        catch e 
            if e isa InterruptException
                rethrow(e)
            end 
            @warn "Failed for coeff[$idx] = $val: $e"
            push!(min_dists, NaN)
        end 
    end
    
    results[idx] = min_dists
end

# Plot
gr()
p = plot(title="Coeff[2]=$drift_lie_coeff fixed, tmax = $(solver.tmax), dt=$(solver.dt)",
         xlabel="Coefficient value", ylabel="Min distance")
for idx in vary_indices
    plot!(p, collect(test_values), replace(results[idx], NaN => missing), label="coeff[$idx]", marker=:circle, yscale=:log10)
end
display(p)
