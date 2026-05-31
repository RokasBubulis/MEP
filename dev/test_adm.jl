include("../src/structs.jl")
include("../src/adm.jl")

using BenchmarkTools

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
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

x_lie = algebra.im_control_lie
y_lie = algebra.neg_im_drift_lie
α = 1.0

res_arr3 = zeros(Float64, length(algebra.lie_basis))
m0 = rand(length(algebra.p_basis))
m0 /= norm(m0)
costate_arr = zeros(Float64, length(algebra.lie_basis))
costate_arr[2:end] = m0
@btime optimal_adjoint_drift_lie_nondiff!(res_arr3, costate_arr, algebra, stor) # ~23 μs
res_arr4 = zeros(Float64, length(algebra.lie_basis))
@btime optimal_adjoint_drift_lie!(res_arr4, costate_arr, algebra, stor) # ~10 μs

println(res_arr3 - res_arr4)