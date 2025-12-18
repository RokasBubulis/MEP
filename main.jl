include("generators.jl")
include("lie_algebra.jl")
include("time_optimal_solver.jl")

# Optimal time calculations
# General
tmin = 1
tmax = 20
print_intermediate = true
coset_hard_tol = 1e-4

function one_qubit_example()
    n_levels = 2
    n_qubits = 1
    turning_point_factor = 1.2

    X = operator(Xop([1]), n_qubits)
    Z = operator(Zop([1]), n_qubits)
    gens = [Z, X]

    target = -im * operator(Yop([1]), n_qubits)

    params = Params(
        -im * X,
        -im * diag(Z),
        n_levels,
        n_qubits,
        tmin,
        tmax,
        turning_point_factor,
        coset_hard_tol,
        print_intermediate,
        Ref(1.0)
    )

    println("One qubit example")
    compute_optimal_time(gens, target, params)
    println("---")
end


function two_qutrit_example()
    n_levels = 3
    n_qubits = 2
    turning_point_factor = 1.3

    gens = construct_Ryd_generators(n_qubits)
    target = construct_CZ_target(n_qubits, n_levels)

    params = Params(
        -im * gens[2],
        -im * diag(gens[1]),
        n_levels,
        n_qubits,
        tmin,
        tmax,
        turning_point_factor,
        coset_hard_tol,
        print_intermediate,
        Ref(1.0)
    )

    println("2 qutrit all symmetric example")
    compute_optimal_time(gens, target, params)
end


one_qubit_example()
two_qutrit_example()

