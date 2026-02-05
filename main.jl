using Roots, BenchmarkTools

include("generators.jl")
include("lie_algebra.jl")
include("time_optimal_solver.jl")

# General parameters
tmin = 0.2*2*pi
tmax = 1.5*2*pi
print_intermediate = true
coset_hard_tol = 1e-4
turning_point_factor = 1.1

function make_params(lie_basis::Vector{SparseMatrixCSC{T,Int}}, V_Ryd::SparseMatrixCSC{T, Int};
                     tmin::Float64, tmax::Float64,
                     turning_point_factor::Float64,
                     coset_hard_tol::Float64,
                     print_intermediate::Bool=false,
                     previous_alpha::Float64=0.0,
                     previous_gamma::Float64=0.0) where {T}

    dim = size(lie_basis[1], 1)
    @assert size(lie_basis[1],2) == dim "H0 must be square"

    H_temp = copy(lie_basis[2])        
    tmp1   = Matrix{T}(undef, dim, dim)
    tmp2   = Matrix{T}(undef, dim, dim)

    return Params{T}(lie_basis[2], Vector(diag(lie_basis[1])), lie_basis[2:end], Vector(diag(V_Ryd)), tmin, tmax,
                     turning_point_factor, coset_hard_tol,
                     print_intermediate, previous_alpha, previous_gamma,
                     dim,
                     H_temp, tmp1, tmp2)
end


########################################################
# One qubit example
function one_qubit_example()
    n_levels = 2
    n_qubits = 1

    # Operators
    X = operator(Xop([1]), n_qubits)
    Z = operator(Zop([1]), n_qubits)
    gens = [Z, X]

    target = -im * operator(Yop([1]), n_qubits)

    # convert target from lab to rotating frame
    positions = [0 0]
    V_Ryd = construct_rydberg_drift(positions, n_levels)

    # Check if target is implementable and construct the basis of orthogonal complement of control subalgebra
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"  # this is not valid in rotating frame!
    display(lie_basis[1])
    display(-im*gens[1])
    # Params
    params = make_params(lie_basis, V_Ryd; tmin, tmax, turning_point_factor, coset_hard_tol, print_intermediate)

    print_intermediate && println("=== One qubit example ===")
    P = compute_optimal_time(target, params, "1_qubit")
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
    V_Ryd = construct_rydberg_drift(positions, n_levels)
    # U_Ryd = exp(-im*Matrix(V_Ryd)*pi)
    # target = sparse(adjoint(U_Ryd) * target)
    #display(target)
    display(V_Ryd)

    # Check if target is implementable and construct the basis of orthogonal complement of control subalgebra
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"  # this is not valid in rotating frame!

    # Params
    params = make_params(lie_basis, V_Ryd; tmin, tmax, turning_point_factor, coset_hard_tol, print_intermediate)


    println("=== Two qutrit example ===")
    P = compute_optimal_time(target, params, "2_qutrit")
end

########################################################
# Two qutrit example
function two_qubit_example()
    n_levels = 2
    n_qubits = 2

    # Generators and target
    drift = operator(Xop([1, 2]), n_qubits)
    control = operator(Zop([1]), n_qubits) + operator(Zop([2]), n_qubits)

    gens = [control, drift]
    target = construct_CZ_target(n_qubits, n_levels)  # lab frame

    # convert target from lab to rotating frame
    positions = [0 0; 0 1]
    V_Ryd = construct_rydberg_drift(positions, n_levels)
    # U_Ryd = exp(-im*Matrix(V_Ryd)*pi)
    # target = sparse(adjoint(U_Ryd) * target)
    #display(target)
    # Check if target is implementable and construct the basis of orthogonal complement of control subalgebra
    lie_basis = construct_lie_basis_general(gens)
    @assert check_if_implementable(lie_basis, target) "Target is not implementable"  # this is not valid in rotating frame!

    # Params
    params = make_params(lie_basis, V_Ryd; tmin, tmax, turning_point_factor, coset_hard_tol, print_intermediate)

    println("=== Two qubit example ===")
    P = compute_optimal_time(target, params, "2_qubit")
    return nothing
end

########################################################
# Run examples
P1 = one_qubit_example()
# P2 = two_qutrit_example()
#one_qubit_example()
#two_qubit_example()

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

