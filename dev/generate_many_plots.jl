include("../src/structs.jl")
include("../src/propagation.jl")
using Kronecker, Plots, LaTeXStrings
include("many_plots_generator.jl")

# generators
n_qubits = 2
tmax = 10
tstar_min = 3
tstar_max = tmax
verbose = false
dt=1e-2

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
println("Setup finished for 2 qubits")

for gate_type in ["k0", "k1"]
    for phase in [pi, pi/2]
        file_name = "new_results/n_$(n_qubits)_phase_$(round(phase, sigdigits=2))_type_$(gate_type)_"
        title_str = latexstring("\\operatorname{Target:}\\;C_$(n_qubits-1)Phase(ϕ/π=$(phase/pi),\\; $gate_type), \\quad t_{optimal}>$(solver.tstar_min), \\quad Δt=$dt")
        generate_save_plots(gate_type, phase, n_qubits, algebra, title_str, file_name)
    end 
end 
println("Plotting finished for 2 qubits")

# generators
n_qubits = 3
tmax = 20
tstar_min = 3
tstar_max = tmax
verbose = false
dt=1e-2

im_control, im_drift = im .* construct_Ryd_generators(n_qubits)
algebra = Algebra(im_control, im_drift)
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-8
unitary_tol = 1e-10
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_min, tstar_max)
println("Setup finished for 3 qubits")

for gate_type in ["k0", "k1"]
    for phase in [pi, pi/2]
        file_name = "new_results/n_$(n_qubits)_phase_$(round(phase, sigdigits=2))_type_$(gate_type)_"
        title_str = latexstring("\\operatorname{Target:}\\;C_$(n_qubits-1)Phase(ϕ/π=$(phase/pi),\\; $gate_type), \\quad t_{optimal}>$(solver.tstar_min), \\quad Δt=$dt")
        generate_save_plots(gate_type, phase, n_qubits, algebra, title_str, file_name)
    end 
end 
println("Plotting finished for 3 qubits")





