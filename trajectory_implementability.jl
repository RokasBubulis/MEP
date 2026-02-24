using LinearAlgebra
using Kronecker
using SparseArrays

# Install MagnusTensor with: 
# ] add https://gitlab.tue.nl/20235021/magnustensor#MagnusTensor-1.1.0

using MagnusTensor
using .LieAlgebraUtils
using .GateUtils
using .AdaptiveOptimization

si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0] 
sx = ComplexF64[1.0 0.0 0.0;0.0 0.0 1.0; 0.0 1.0 0.0]
sz = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 -1.0]
k0 = ComplexF64[1.0; 0.0; 0.0]
k1 = ComplexF64[0.0; 1.0; 0.0]

q = ComplexF64[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
function hamiltonian(n)
    hx = 0.5*sum([ ⊗([ (i==j ? sx : q) for j ∈ 1:n]...) for i ∈ 1:n ]) |> sparse
    hz = 0.5*sum([ ⊗([ (i==j ? sz : si) for j ∈ 1:n]...) for i ∈ 1:n ]) |> sparse

    return hx, hz
end

function construct_trajectory(phi, n)
    states_input = [ [⊗([ ( j<=i ? k1 : k0 ) for j in 1:n  ]...) |> vec |> sparse for i ∈ 0:n]..., [⊗([ ( j==i ? k1 : k0 ) for j in 1:n  ]...) |> vec |> sparse for i ∈ 1:n]...]

    gate = ⊗([si for _ ∈ 1:n]...) *exp(-1im*phi)  + ⊗([k1*k1' for _ ∈ 1:n]...)*(1-exp(-1im*phi)) |> sparse
    states_output = [ gate*ψ for ψ ∈ states_input ]

    return OptimizationTrajectory(states_input, states_output)
end

hx, hz = hamiltonian(2)

lie = generate_algebra_fast([hx, hz], 8, trim_atol=1e-15, op_rtol=1e-5)
## Setting up the problem

gatelie = GateConstant(lie; init=:random)

circuit = GateSequence([gatelie])
phi = π
nqubits = 2 

trajectory = construct_trajectory(phi, nqubits)

# validating gradients
avg, sig = AdaptiveOptimization.validate_gradients(circuit, Vector{GateDynamicHermite}(), Vector{Vector{Integer}}(), trajectory, 5, 1e-7, lambda_magnus=1.0, verbose=0)

##

history = Vector{AbstractGateState}()

res = optimize_circuit(circuit, Vector{Int}(), Vector{GateDynamicHermite}(), trajectory, storeStates=history, show_trace=true, show_every=5, time_limit=300, grad_tolerance=50*1e-6, validate_grads=false)

## Retrieving the Lie algebra element


lie_op = hx*0

for (μ, Lμ) in enumerate(lie.orthogonal_elements)
    lie_op += gatelie.controls[μ]*gatelie.controls_herm_coef[μ] * Lμ
end
# you can check ishermitian(lie_op)

display(lie_op)
