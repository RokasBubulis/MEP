include("../src/generators.jl")
include("../src/lie_algebra.jl")

isometries = build_Rydberg_isometries(2)
display(isometries[1])
display(isometries[2])