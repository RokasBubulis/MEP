using LinearAlgebra
using Optim
using Plots
using Random

include("generators.jl")

function H_α!(H_α::AbstractMatrix, drift::AbstractMatrix, diag_control::Vector, α::Float64)
    n = size(drift, 1)
    @inbounds for i in 1:n, j in 1:n
        H_α[i,j] = drift[i,j] * exp(α * (diag_control[j] - diag_control[i]))
    end
    return nothing
end

struct Params
    drift::Matrix{ComplexF64}
    diag_control::Vector{ComplexF64}
    H_alpha::Matrix{ComplexF64}
end

function H_optimal!(M::AbstractMatrix, params::Params)
    drift, l = params.drift, params.diag_control
    n = size(drift, 1)

    function f(x)
        α = x[1]
        func = 0.0
        @inbounds for i in 1:n, j in 1:n
            Δ = l[j] - l[i]
            e = exp(α * Δ)
            z = drift[i,j] * M[j,i]
            func += real(e*z)
        end
        return -func
    end

    function g!(G, x)
        α = x[1]
        grad = 0.0
        @inbounds for i in 1:n, j in 1:n
            Δ = l[j] - l[i]
            e = exp(α * Δ)
            z = drift[i,j] * M[j,i]
            grad += real(Δ * e * z)
        end
        G[1] = -grad
        return nothing
    end

    res = optimize(f, g!, [-Float64(π/2)], [Float64(π/2)], [0.0], Fminbox(BFGS()))
    α_opt = Optim.minimizer(res)[1]
    H_α!(params.H_alpha, drift, l, α_opt)

    return α_opt, res
end

function build_M!(M::AbstractMatrix{T}, m::AbstractVector{Float64}, p_basis::Vector{SparseMatrixCSC{T, Int}})
    fill!(M, zero(T))
    for (i, m_coeff) in enumerate(m)
        M .+= m_coeff * p_basis[i]
    end
    return nothing
end 

n_qubits = 2
control, drift = construct_Ryd_generators(n_qubits)
diag_control = diag(control)
gens = [control, drift]
lie_basis = construct_lie_basis_general(gens)
p_basis = lie_basis[2:end]
M = zeros(ComplexF64, size(p_basis[1])...)
build_M!(M, randn(length(p_basis)), p_basis)

display(drift)
display(control)
display(M)

params = Params(-im*drift, -im*diag_control, zeros(ComplexF64, size(drift, 1), size(drift, 1)))

α_opt, res = H_optimal!(M, params)

println("α_opt = ", α_opt)
println(res)

function objective_scalar(α, M, params)
    drift, l = params.drift, params.diag_control
    n = size(drift, 1)

    func = 0.0
    @inbounds for i in 1:n, j in 1:n
        Δ = l[j] - l[i]
        e = exp(α * Δ)
        z = drift[i,j] * M[j,i]
        func += real(e*z)
    end
    return func
end

α_grid = range(-π/2, π/2, length=400)
vals = [objective_scalar(α, M, params) for α in α_grid]

plot(α_grid, vals, label="objective")
vline!([α_opt], label="α_opt", linestyle=:dash)