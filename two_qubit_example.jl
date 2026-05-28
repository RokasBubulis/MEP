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

##

m0 = rand(length(algebra.p_basis))
# solver parameters
tmax = 10
tol = 1e-10

dt_rk = 1e-4
solver = SolverParams(tmax, dt_rk, tol)
stor_rk = Storage{ComplexF64}(dim, length(algebra.lie_basis))
Us_rk = propagate_RK4(m0, algebra, system, solver, stor_rk, save_only_U=true)[end]
