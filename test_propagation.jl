using Plots
gr()

include("generators.jl")
include("lie_algebra.jl")
include("set_params.jl")
include("propagation.jl")

params = prepare_trivial_2D_setup()
beta = 0.0
gamma = π
params.system_params.target = sparse(exp(-im*beta*Matrix(construct_YQ_target(2)) 
                                        - gamma*Matrix(params.derived_args.p_basis[1])))

m0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ts, Us, Ms, dists, Hs = propagate(m0, params)
ts /= π
println("Minimum distance $(minimum(dists)) at time $(ts[argmin(dists)]) π")
n = size(params.system_params.im_control, 1)
Ms_overlap_with_drift = [abs(tr(M * adjoint(params.derived_args.p_basis[1]))) for M in Ms]
Us_overlap_with_target = [1/n * abs(tr(U * adjoint(params.system_params.target))) for U in Us]
Hs_overlap_with_M = [abs(tr(Hs[i] * Ms[i])) for i in eachindex(ts)]
@assert isapprox(Ms[1], params.derived_args.p_basis[1])

p1 = plot(ts, dists, title="min dist=$(round(minimum(dists), sigdigits=3)) at t=$(round(ts[argmin(dists)], sigdigits=3)) π")
xlabel!(p1, "t (π)")
ylabel!(p1, "d_targ_coset")
p2 = plot(ts, Ms_overlap_with_drift)
xlabel!(p2, "t (π)")
ylabel!(p2, "M(t) overlap with drift")
p3 = plot(ts, Us_overlap_with_target)
xlabel!(p3, "t (π)")
ylabel!(p3, "U(t) overlap with U_T")
p4 = plot(ts, Hs_overlap_with_M)
xlabel!(p4, "t (π)")
ylabel!(p4, "M(t) overlap with adjoint drift")

plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
#savefig(plt, "results/drift_propagation_test.png")
display(plt)
