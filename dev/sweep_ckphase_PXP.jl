# Same sweep as dev/sweep_ckphase.jl, with the system set up by construct_PXP_generators.
# Output: res/PXP_T_vs_phase_success.png, res/PXP_alpha_controls.png, res/PXP_T_vs_phase.jld2, res/PXP_T_vs_phase.txt
# Run:  julia -t auto --project=. dev/sweep_ckphase_PXP.jl      (Ctrl+C stops it)

include("../src/structs.jl")
include("../src/propagation.jl")
include("../src/optimise.jl")

generator_fn = construct_PXP_generators
system_name  = "PXP"
include("sweep_ckphase.jl")
