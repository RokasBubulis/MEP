include("structs.jl")
include("optimisation.jl")
include("../liepmp/src/unwrap_evolution.jl")

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# prepare system struct 
target = Matrix(SparseMatrixCSC{ComplexF64, Int}(I, dim, dim))
target[5,5] = -1.0
title="CZ"
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
system = System{ComplexF64}(im_control, im_drift, target, target_logic)
stor = Storage{ComplexF64}(size(im_control,1), length(algebra.lie_basis))
##

using LinearAlgebra
using Plots

include("test.jl")

alphas = range(-40.0, 40.0, length=200)
norms  = zeros(length(alphas))
Y = -system.im_drift
y_lie = project_algebra(Y, algebra)
res_arr = zeros(length(algebra.lie_basis))

for (i, alpha) in enumerate(alphas)
    X = -alpha * system.im_control

    x_lie = project_algebra(system.im_control, algebra)
    adjoint_drift_new!(res_arr, alpha, x_lie, y_lie, algebra.structure_tensor, stor)

    local res = sum(res_arr[i] .* algebra.lie_basis[i] for i in eachindex(res_arr))

    res_true = exp(Matrix(X)) * Y * exp(-Matrix(X))

    norms[i] = norm(res_true - res)
end

plot(alphas, norms,
    xlabel = "alpha",
    ylabel = "norm",
    title  = "adjoint drift error",
    yscale=:log10,
    legend = false)
