using BenchmarkTools

include("../src/structs.jl")
include("../src/propagation.jl")

im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
tar = TargetContainer(target_logic)
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
adaptive=false
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol)
println("Setup finished")


# Build a mock integrator struct that mimics what DifferentialEquations passes in
struct MockIntegrator
    p::Tuple
    dt::Float64
    t::Float64
end

m0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Reconstruct the exact inputs propagate would use
m0_arr = zeros(Float64, length(algebra.lie_basis))
m0_arr[2:end] = m0
u = m0_arr
stor.U .= stor.U0

tracker = OutputTracker(1.0, solver.tmax)
mock_integrator = MockIntegrator(
    (algebra, tar, solver, stor, tracker),
    1e-4,   # representative dt
    0.0     # t = 0
)

# Define step_update standalone (copy exactly from propagate, no closure)
function step_update_standalone(u, t, integrator)
    tar, solver, stor, tracker = integrator.p[2], integrator.p[3], integrator.p[4], integrator.p[5]

    optimal_adjoint_drift_lie!(stor.H_opt_lie, u, algebra, stor)
    Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra)
    copyto!(stor.dU, stor.H_opt)
    lmul!(integrator.dt, stor.dU)
    exponential!(stor.dU, stor.exp_method, stor.exp_cache)
    mul!(stor.U_buffer, stor.dU, stor.U)
    stor.U .= stor.U_buffer
    check_unitarity(stor.U, stor.U_unitary_buffer_check, solver.unitary_tol)

    dist = distance(stor.U, tar, algebra, stor)
    if dist < 0.0
        dist = abs(dist)
    end
    if dist < tracker.min_dist
        tracker.min_dist = dist
        tracker.tstar = t
    end
end

using Profile, PProf  # or just use @allocated per line

# Isolate each line individually
println("optimal_adjoint_drift_lie!: ", 
    @allocated optimal_adjoint_drift_lie!(stor.H_opt_lie, u, algebra, stor))

println("Lie_to_Hilbert!: ", 
    @allocated Lie_to_Hilbert!(stor.H_opt, stor.H_opt_lie, algebra))

println("copyto!: ", 
    @allocated copyto!(stor.dU, stor.H_opt))

println("lmul!: ", 
    @allocated lmul!(mock_integrator.dt, stor.dU))

println("exponential!: ", 
    @allocated exponential!(stor.dU, stor.exp_method, stor.exp_cache))

println("mul!: ", 
    @allocated mul!(stor.U_buffer, stor.dU, stor.U))

println("stor.U .= stor.U_buffer: ", 
    @allocated (stor.U .= stor.U_buffer))

println("check_unitarity: ", 
    @allocated check_unitarity(stor.U, stor.U_unitary_buffer_check, solver.unitary_tol))

println("distance: ", 
    @allocated distance(stor.U, tar, algebra, stor))
    
# Benchmark
@benchmark step_update_standalone($u, 0.0, $mock_integrator)

# For allocation count specifically:
#@allocated step_update_standalone(u, 0.0, mock_integrator)