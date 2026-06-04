using ExponentialUtilities, BenchmarkTools

a = rand(ComplexF64, 4, 4)
b = copy(a)

mem = Matrix{ComplexF64}(undef, 4, 4)
method = ExpMethodHigham2005();
cache = ExponentialUtilities.alloc_mem(a, method); 

a_orig = copy(a)

@btime begin
    copyto!($a, $a_orig)
    exponential!($a, $method, $cache)
end
@btime exp($a_orig)

# for _ in 1:5
#     a = randn(ComplexF64, 4, 4)
#     b = copy(a)
#     exponential!(a, method, cache)
#     res = exp(b)
#     println(norm(a - res))
# end 



