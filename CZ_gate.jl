using QuantumOptics, LinearAlgebra, Printf

args = Dict(
    "pulse_time" => 2*pi*1.215,
    "phase_amplitude" => 2*pi*0.1122,
    "modulation_freq" => 1.0431,
    "phase_offset" => -0.7318,
    "linear_detuning" => 0,
    "rabi_amplitude" => 2*pi*4.6*1e6, 
    "blockade_strength" => 2*pi*450*1e6
)
basis = SpinBasis(1//2)
function global_rabi_pulse(t)
global_phase = args["phase_amplitude"] * cos(args["modulation_freq"] * t - args["phase_offset"]) + args["linear_detuning"] * t
global_rabi_pulse = exp(im*global_phase)
end

function H_single(t, psi)
    omega = global_rabi_pulse(t)
    H = [0 omega/2; conj(omega)/2 0]
    return Operator(basis, H)
end

function H_full(t, psi)
    single_H = H_single(t, psi)
    interaction_term = zeros(4,4)
    interaction_term[4,4] = args["blockade_strength"] / args["rabi_amplitude"]
    I = Operator(basis, [1 0; 0 1])
    basis2 = tensor(basis, basis)
    tensor(I, single_H) + tensor(single_H, I) + Operator(basis2, interaction_term)
end

function cz_gate(args)
    basis = SpinBasis(1//2)
    cz_uncalibrated = zeros(ComplexF64, 4, 4)
    cz_uncalibrated[1,1] = 1.0
    tspan = [0:0.01:args["pulse_time"];]

    # Evolve |1>
    psi0_1 = spinup(basis)
    tout, psi_final_1 = timeevolution.schroedinger_dynamic(tspan, psi0_1, H_single)
    # Evolve |11>
    psi0_2 = tensor(spinup(basis), spinup(basis))
    tout, psi_final_2 = timeevolution.schroedinger_dynamic(tspan, psi0_2, H_full)

    cz_uncalibrated[2,2] = psi_final_1[end].data[2]
    cz_uncalibrated[3,3] = psi_final_1[end].data[2]
    cz_uncalibrated[4,4] = psi_final_2[end].data[4]

    # println(round.(cz_uncalibrated, digits=2))

    # Single qubit phase calibration
    phi = angle(cz_uncalibrated[2,2])
    Z_phi = zeros(ComplexF64, 2, 2)
    Z_phi[1,1] = 1.0 
    Z_phi[2,2] = exp(-1im * phi)
    calibration_operator = kron(Z_phi, Z_phi)
    calibration_operator * cz_uncalibrated
end

gate = cz_gate(args)
for i in 1:size(gate,1)
    for j in 1:size(gate,2)
        @printf("%6.3f + %6.3fi  ", real(gate[i,j]), imag(gate[i,j]))
    end
    println()
end
