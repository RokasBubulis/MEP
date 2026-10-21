
include("adm.jl")
include("sensitivity.jl")

function ode_problem!(du, u, p, t)
    M, χ, B = u.x
    dM, dχ, dB = u.dx

    algebra, stor = p  # TODO extend the parameters?
    # costate propagation
    # obtain H_opt 
    optimal_adjoint_drift_lie_analytic!(stor.H_opt_lie, M, algebra, stor)
    # dM/dt = [H_opt, M] in Lie space; dim = dx1
    lie_bracket_coeffs!(dM, algebra.structure_tensor, stor.H_opt_lie, M)

    # costate sensitivity propagation
    # dχ/dt = Dχ in Lie space; dim = dxd
    costate_sensitivity_drift!(stor.D, algebra, stor, M)
    mul!(dχ, stor.D, χ)

    # state sensitivity propagation
    # obtain E 
    state_sensitivity_source!(stor.E, algebra, stor)
    # obtain F 
    state_sensitivity_drift!(stor.F, algebra, stor)
    # dB/dt = Eχ + FB 
    mul!(dB, stor.E, χ)
    mul!(dB, stor.F, B, 1, 1)  # mul!(A,B,C,a,b) = A -> a*B*C + b*A
end 