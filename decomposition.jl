using LinearAlgebra, SparseArrays

ket0 = ComplexF64[1,0,0]; ket1 = ComplexF64[0,1,0]; ketr = ComplexF64[0,0,1]

P0 = ket0 * ket0'        
P1 = ket1 * ket1'        
Pr = ketr * ketr'        

function Q()
    sparse(P0 + P1)
end

function R()
    sparse(Pr)
end

function X()
    sparse(ket0*ket1' + ket1*ket0' + Pr)
end

function XR()
    sparse(P0 + ket1*ketr' + ketr*ket1')
end

function P0op()
    sparse(P0)
end

I3 = sparse(Matrix{ComplexF64}(I, 3, 3))

function embed(op::SparseMatrixCSC, site::Int)
    op = SparseMatrixCSC{ComplexF64,Int}(op) 
    if site == 1
        return kron(op, I3)
    elseif site == 2
        return kron(I3, op)
    else
        error("only two sites supported")
    end
end

# global two site operators
Q1  = embed(Q(), 1);   Q2  = embed(Q(), 2)
R1  = embed(R(), 1);   R2  = embed(R(), 2)
X1  = embed(X(), 1);   X2  = embed(X(), 2)
XR1 = embed(XR(), 1);  XR2 = embed(XR(), 2)
P01 = embed(P0op(), 1); P02 = embed(P0op(), 2)

basis_ops = Dict(
    "Q1" => Q1, "Q2" => Q2,
    "R1" => R1, "R2" => R2,
    "X1" => X1, "X2" => X2,
    "XR1" => XR1, "XR2" => XR2,
    "P01" => P01, "P02" => P02
)

function decompose(op, basis::Dict{String,SparseMatrixCSC{ComplexF64, Int64}})
    coeffs = Dict{String,ComplexF64}()
    for (name, b) in basis
        coeffs[name] = tr(b' * op)
    end
    return coeffs
end

# for (idx, O) in enumerate(im*lie_basis)
#     println("Decomposition of element $idx of hermitian basis")
#     pretty_table(O)
#     coeffs = decompose(O, basis_ops)
#     for (name, c) in coeffsfjjdhfd
#         println("$name : $c")
#     end
#     println("---")
# end
# println("Decomposition of input density matrix")
# coeffs_input = decompose(convert(SparseMatrixCSC{ComplexF64, Int64}, input_matrix), basis_ops)
# for (name, c) in coeffs_input
#     println("$name : $c")
# end