using Roots, BenchmarkTools

include("generators.jl")
include("lie_algebra.jl")
include("time_optimal_solver.jl")

# General parameters
tmin = 0.1*2*pi
tmax = 2*2*pi
print_intermediate = true
coset_hard_tol = 1e-4
turning_point_factor = 1.1

########################################################
# One qubit example
function one_qubit_example()
    n_levels = 2
    n_qubits = 1

    # Operators
    X = operator(Xop([1]), n_qubits)
    Z = operator(Zop([1]), n_qubits)
    gens = [Z, X]

    target = -1im * operator(Yop([1]), n_qubits)

    # convert target from lab to rotating frame
    positions = [0 0]
    V_Ryd = construct_rydberg_drift(positions; n_levels)

    # Params
    params = make_params(-im*X, Vector(-im*diag(Z)), V_Ryd,
        tmin=tmin, tmax=tmax,
        turning_point_factor=turning_point_factor,
        coset_hard_tol=coset_hard_tol,
        print_intermediate=true
    )

    print_intermediate && println("=== One qubit example ===")
    P = compute_optimal_time(gens, target, params, "1_qubit")
end

########################################################
# Two qutrit example
function two_qutrit_example()
    n_levels = 3
    n_qubits = 2

    # Generators and target
    gens = construct_Ryd_generators(n_qubits)  # rotating frame
    target = construct_CZ_target(n_qubits, n_levels)  # lab frame
    #target -= 7/9*I

    # convert target from lab to rotating frame
    positions = [0 0; 0 1]
    V_Ryd = construct_rydberg_drift(positions; n_levels)
    # U_Ryd = exp(-im*Matrix(V_Ryd)*pi)
    # target = sparse(adjoint(U_Ryd) * target)
    #display(target)
    display(V_Ryd)

    # Params
    params = make_params(-im*gens[2], Vector(-im*diag(gens[1])), V_Ryd, 
        tmin=tmin, tmax=tmax,
        turning_point_factor=turning_point_factor,
        coset_hard_tol=coset_hard_tol,
        print_intermediate=true
    )

    println("=== Two qutrit example ===")
    P = compute_optimal_time(gens, target, params, "2_qutrit")
end

########################################################
# Run examples
P1 = one_qubit_example()
P2 = two_qutrit_example()

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

