using QuantumOptics
basis = SpinBasis(1//2)
ψ₀ = spindown(basis)
const sx = sigmax(basis)
function H_pump(t, psi)
  return sin(t)*sx
end
tspan = [0:0.1:10;]
tout, ψₜ = timeevolution.schroedinger_dynamic(tspan, ψ₀, H_pump)