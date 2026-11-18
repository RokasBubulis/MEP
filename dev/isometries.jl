include("../src/generators.jl")
include("../src/lie_algebra.jl")

n_qubits = 2
isometries = build_Rydberg_isometries(n_qubits)
im_control, im_drift = construct_HJB_generators(n_qubits)

for λ in 1:n_qubits
    println("λ = $(λ)")
    display(adjoint(isometries[λ]) * im_control * isometries[λ])
    display(adjoint(isometries[λ]) * im_drift * isometries[λ])
end 