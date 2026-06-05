include("../src/structs.jl")
include("../src/adm.jl")

using BenchmarkTools

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# prepare system struct 
target_logic = SparseMatrixCSC{ComplexF64, Int}(I, 4, 4)
target_logic[4,4] = -1.0
tar = TargetContainer(target_logic)
stor = Storage{ComplexF64}(dim, length(algebra.lie_basis))

x_lie = algebra.im_control_lie
y_lie = algebra.neg_im_drift_lie
α = 1.0

res_arr3 = zeros(Float64, length(algebra.lie_basis))
res_arr4 = zeros(Float64, length(algebra.lie_basis))

##

for i in 1:100
    m0 = rand(length(algebra.p_basis))
    m0 /= norm(m0)
    costate_arr = zeros(Float64, length(algebra.lie_basis))
    costate_arr[2:end] = m0
    res_arr3 .= 0
    res_arr4 .= 0
    stor.alpha = 0
    optimal_adjoint_drift_lie_analytic!(res_arr3, costate_arr, algebra, stor)
    stor.alpha = 0
    optimal_adjoint_drift_lie_optimiser!(res_arr4, costate_arr, algebra, stor)
    nrm = norm(res_arr3 - res_arr4)
    if nrm > 1e-8
        @warn "$i : norm= $nrm"
    end 
end 
println("analytic")
@btime begin 
    stor.alpha = 0
    optimal_adjoint_drift_lie_analytic!(res_arr4, costate_arr, algebra, stor) # ~10 μs, 133 allocs
end 
println("optimiser")
@btime begin 
    stor.alpha = 0
    optimal_adjoint_drift_lie_optimiser!(res_arr3, costate_arr, algebra, stor)
end 

# example output: 
# ┌ Warning: 55 : norm= 1.0689562827990031e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:56
# ┌ Warning: 68 : norm= 1.8561914723068352e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:56
# analytic
#   16.211 μs (127 allocations: 8.14 KiB)
# optimiser
#   37.917 μs (370 allocations: 23.53 KiB)