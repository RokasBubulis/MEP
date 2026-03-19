using CMAEvolutionStrategy

include("propagation.jl")

function objective(m, params)
    dist, tstar =  propagate(m, params)
    return dist
end

params = prepare_trivial_2D_setup()
beta = 1.0
gamma = 0.1
params.system_params.target = sparse(exp(
    - beta * Matrix(params.system_params.im_drift) 
    - im *gamma * Matrix(construct_YQ_target(2))
    ))

m0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = minimize((m)->objective(m, params), m0, 0.5; maxiter=50, verbosity=1)
m_best = xbest(result)

dist, tstar = propagate(m_best, params)
println("Lowest distance $(dist) at time $tstar")
