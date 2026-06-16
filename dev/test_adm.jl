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
stor = Storage(algebra.n_particles, length(algebra.lie_basis))

x_lie = algebra.im_control_lie
y_lie = algebra.neg_im_drift_lie
α = 1.0

res_arr3 = zeros(Float64, length(algebra.lie_basis))
res_arr4 = zeros(Float64, length(algebra.lie_basis))

##
vals = []
m0 = rand(length(algebra.p_basis))
m0 /= norm(m0)
costate_arr = zeros(Float64, length(algebra.lie_basis))
costate_arr[2:end] = m0
αlist = range(-10, 10, 100)
for α in αlist 
    obj = adjoint_drift_obj(α, costate_arr, algebra, stor)
    push!(vals, obj)
end 

vals_vec = Float64.(vals)
troughs = findall(i -> vals_vec[i] < vals_vec[i-1] && vals_vec[i] < vals_vec[i+1], 
                  2:length(vals_vec)-1) .+ 1

p = plot(αlist, vals_vec)
scatter!(p, αlist[troughs], vals_vec[troughs], 
         series_annotations=text.(round.(αlist[troughs], digits=3), :top, 8),
         label="troughs", markershape=:dtriangle)

display(p)

##

for i in 1:200
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

costate_arr = zeros(Float64, length(algebra.lie_basis))
costate_arr[2:end] = m0
@btime g, h = adjoint_drift_obj_derivatives(α, costate_arr, algebra, stor)
println("analytic")
@btime begin 
    stor.alpha = 0
    optimal_adjoint_drift_lie_analytic!(res_arr4, costate_arr, algebra, stor)
end 
println("optimiser")
@btime begin 
    stor.alpha = 0
    optimal_adjoint_drift_lie_optimiser!(res_arr3, costate_arr, algebra, stor)
end 

# example output (note: worse than irl as no warm-start)
# ┌ Warning: 3 : norm= 1.5347428753897965e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
# ┌ Warning: 29 : norm= 1.1318724787655718e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
# ┌ Warning: 43 : norm= 1.4164445704395158e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
# ┌ Warning: 44 : norm= 1.174096386074737e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
# ┌ Warning: 128 : norm= 1.55685958252071e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
# ┌ Warning: 137 : norm= 1.9435728747738652e-8
# └ @ Main ~/workspaces/MScProject/dev/test_adm.jl:41
#   1.780 μs (18 allocations: 1.12 KiB)
# analytic
#   15.500 μs (154 allocations: 10.03 KiB)
# optimiser
#   38.601 μs (466 allocations: 29.75 KiB)