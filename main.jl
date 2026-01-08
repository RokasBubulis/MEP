using Roots

include("generators.jl")
include("lie_algebra.jl")
include("time_optimal_solver.jl")

# General parameters
tmin = 1.0
tmax = 10.0
print_intermediate = true
coset_hard_tol = 1e-4

########################################################
# One qubit example
function one_qubit_example()
    n_levels = 2
    n_qubits = 1
    turning_point_factor = 1.2

    # Operators
    X = operator(Xop([1]), n_qubits)
    Z = operator(Zop([1]), n_qubits)
    gens = [Z, X]

    target = -1im * operator(Yop([1]), n_qubits)

    # Params
    params = make_params(-im*X, -im*diag(Z);
        n_levels=n_levels, n_qubits=n_qubits,
        tmin=tmin, tmax=tmax,
        turning_point_factor=turning_point_factor,
        coset_hard_tol=coset_hard_tol,
        print_intermediate=true
    )

    println("=== One qubit example ===")
    compute_optimal_time(gens, target, params)
    println("-------------------------")
end

########################################################
# Two qutrit example
# function two_qutrit_example()
#     n_levels = 3
#     n_qubits = 2
#     turning_point_factor = 1.1

#     # Generators and target
#     gens = construct_Ryd_generators(n_qubits)
#     target = construct_CZ_target(n_qubits, n_levels)

#     # Params
#     params = Params(
#         -1im * gens[2],
#         -1im * diag(gens[1]),
#         n_levels,
#         n_qubits,
#         tmin,
#         tmax,
#         turning_point_factor,
#         coset_hard_tol,
#         print_intermediate,
#         Ref(1.0),
#         -1.0,
#         1.0
#     )

#     println("=== Two qutrit example ===")
#     compute_optimal_time(gens, target, params)
#     println("--------------------------")
# end

########################################################
# Run examples
one_qubit_example()
# two_qutrit_example()

# n_qubits = 1
# X = operator(Xop([1]), n_qubits)
# Z = operator(Zop([1]), n_qubits)
# gens = [Z, X]
# lie_basis = construct_lie_basis_general(gens)
# p_basis = lie_basis[2:end] 
# m = rand(length(p_basis)) .*2 .-1
# M = build_M(m, p_basis)
# H0 = -1im * X
# l = -1im * diag(gens[1])

# function find_alpha(α)
#     n = size(H0, 1)
#     s = 0
#     for j in 2:n, i in 1:j-1
#         Δ = l[j] - l[i]
#         s += real(Δ*exp(α*Δ)*H0[i,j]*M[j,i])
#     end
#     return s
# end

# α_star = find_zero(find_alpha, (-1, 1)) 
# println(α_star)
# println(find_alpha(α_star))

