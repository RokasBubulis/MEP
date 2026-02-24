include("generators.jl")

function construct_CZ_target(n_qubits::Int, n_levels::Int)
    dim = n_levels^n_qubits
    U = SparseMatrixCSC{ComplexF64}(I, dim, dim)  # true identity
    U[5, 5] = -1 + 0im
    return U
end

function check_group_implementability(target::SparseMatrixCSC{float_type, Int}, lie_basis::Vector{SparseMatrixCSC{float_type, Int}})
    choices= 0
    n = size(target, 1)
    combs = [vcat(collect(t), 0) for t in Iterators.product(ntuple(_ -> choices, n-1)...)]
    #combs = [collect(t) for t in Iterators.product(ntuple(_ -> choices, n)...)]
    # lie basis is orthonormal
    θ = angle.(diag(target))
    min_residual = Inf
    best_residual_matrix = similar(target)
    for comb in combs

        phases = θ .+ 2*pi .*comb
        ln_target = spdiagm(0=> im.*phases)
        ln_target .-= (tr(ln_target)/n) * spdiagm(0 => ones(float_type, n))   

        reconstructed_target = spzeros(float_type, size(target)...)
        for i in eachindex(lie_basis)
            coeff = dot(lie_basis[i], ln_target)
            reconstructed_target .+= coeff .* lie_basis[i]
        end

        residual_norm = norm(reconstructed_target - ln_target)
        if residual_norm < min_residual
            min_residual = residual_norm
            best_residual_matrix = reconstructed_target - ln_target
        end
        # min_residual = min(min_residual, residual_norm)

        if residual_norm < 1e-6
            println("Implementability check passed")
            return true
        end
    end
    display(best_residual_matrix)
    println("Implementability check failed with minimum residual: $min_residual")
    return false
end