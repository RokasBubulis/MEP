include("../src/structs.jl")
include("../src/propagation.jl")
include("make_plot.jl")
using Kronecker, ProgressMeter, JLD2

# generators
im_control, im_drift = im .* construct_Ryd_generators(2)
dim = size(im_control, 1)

# prepare Lie algebra struct 
algebra = Algebra(im_control, im_drift)

# gate
k0 = ComplexF64[1.0; 0.0; 0.0]
k1 = ComplexF64[0.0; 1.0; 0.0]
si = ComplexF64[1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
gate_k0(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k0*k0' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse
gate_k1(n, phi) = (⊗([si for _ ∈ 1:n]...) * exp(-1im*phi) + ⊗([k1*k1' for _ ∈ 1:n]...) * (1 - exp(-1im*phi))) |> sparse

function build_tar(type, phi, n)
    stor = Storage(n, length(algebra.lie_basis))
    if type == "k0"
        ryd_target = gate_k0(n, phi)
    elseif type == "k1"
        ryd_target = gate_k1(n, phi)
    else
        throw("unknown gate type")
    end 
    # display(round.(ryd_target, sigdigits=2))
    # target = Matrix{ComplexF64}(undef, 2^n, 2^n)
    # project_from_3_to_2_levels!(target, ryd_target, algebra, stor)
    # display(target)
    # return TargetContainer(target)
    return TargetContainer(ryd_target)
end 
# params
tmax = 10
reltol = 1e-8
abstol = 1e-8
dist_tol = 1e-8
grad_tol = 1e-9
unitary_tol = 1e-10
tstar_threshold = 3
solver = SolverParams(tmax, reltol, abstol, dist_tol, grad_tol, unitary_tol, tstar_threshold)
println("Setup finished")
##
m0 = rand(Float64, length(algebra.p_basis))
m0 /= norm(m0)
dt = 1e-2
optimise_for_phi = 0.5 * pi
gate_type = "k0"
tar = build_tar(gate_type, optimise_for_phi, 2)
m_best, m_log = find_best_initial_costate_with_logging(tar, algebra, solver, stor, dt=dt, m0=m0, show_trace=true, iterations=2000)

##
results = []
for (i, m) in enumerate(m_log) 

    stor = Storage(2, length(algebra.lie_basis))
    d, tstar, Ulist = propagate(m, algebra, solver, stor, tar, save_ulist=true, dt=dt)
    n = length(Ulist)
    ts = dt .* (1:n)
    dists = [distance(Ulist[i], tar, algebra, stor) for i in eachindex(ts)]
    push!(results, (i=i, m=m, d=d, tstar=tstar, Ulist=Ulist, dists=dists, ts=ts))
end 

##
t_min_idx = 1
t_max_idx = 1000
d_arr      = [r.d       for r in results]
ts_arr     = [r.ts[t_min_idx:t_max_idx]      for r in results]
dists_arr  = [r.dists[t_min_idx:t_max_idx]   for r in results]
i_arr = [r.i for r in results]

D = hcat([log10.(v) for v in dists_arr]...)
p = contourf(i_arr, ts_arr[1], D;
    colorbar=true,
    c=:viridis,
    xlabel="Objective calls",
    ylabel="t",
    colorbar_title="log(d(t))",
    size=(800, 500),
    levels=50,
    title="gate: CPhase(ϕ), m0 optimised for ϕ/π=$(optimise_for_phi/pi), $gate_type, t*>$(solver.tstar_threshold)"

    )
display(p)