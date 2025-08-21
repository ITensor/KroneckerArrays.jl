using LinearAlgebra:
  LinearAlgebra,
  Diagonal,
  Eigen,
  SVD,
  det,
  diag,
  eigen,
  eigvals,
  lq,
  mul!,
  norm,
  qr,
  svd,
  svdvals,
  tr

using LinearAlgebra: LinearAlgebra
function KroneckerArray(J::LinearAlgebra.UniformScaling, ax::Tuple)
  return Eye{eltype(J)}(arg1.(ax)) ⊗ Eye{eltype(J)}(arg2.(ax))
end
function Base.copyto!(a::KroneckerArray, J::LinearAlgebra.UniformScaling)
  copyto!(a, KroneckerArray(J, axes(a)))
  return a
end

using LinearAlgebra: LinearAlgebra, pinv
function LinearAlgebra.pinv(a::KroneckerArray; kwargs...)
  return pinv(a.a; kwargs...) ⊗ pinv(a.b; kwargs...)
end

function LinearAlgebra.diag(a::KroneckerArray)
  return copy(DiagonalArrays.diagview(a))
end

# Allows customizing multiplication for specific types
# such as `Eye * Eye`, which doesn't return `Eye`.
function _mul(a::AbstractArray, b::AbstractArray)
  return a * b
end

function Base.:*(a::KroneckerArray, b::KroneckerArray)
  return _mul(a.a, b.a) ⊗ _mul(a.b, b.b)
end

function LinearAlgebra.mul!(
  c::KroneckerArray, a::KroneckerArray, b::KroneckerArray, α::Number, β::Number
)
  iszero(β) ||
    iszero(c) ||
    throw(
      ArgumentError(
        "Can't multiple KroneckerArrays with nonzero β and nonzero destination."
      ),
    )
  mul!(c.a, a.a, b.a)
  mul!(c.b, a.b, b.b, α, β)
  return c
end

function LinearAlgebra.tr(a::KroneckerArray)
  return tr(a.a) ⊗ tr(a.b)
end

function LinearAlgebra.norm(a::KroneckerArray, p::Int=2)
  return norm(a.a, p) ⊗ norm(a.b, p)
end

# Matrix functions
const MATRIX_FUNCTIONS = [
  :exp,
  :cis,
  :log,
  :sqrt,
  :cbrt,
  :cos,
  :sin,
  :tan,
  :csc,
  :sec,
  :cot,
  :cosh,
  :sinh,
  :tanh,
  :csch,
  :sech,
  :coth,
  :acos,
  :asin,
  :atan,
  :acsc,
  :asec,
  :acot,
  :acosh,
  :asinh,
  :atanh,
  :acsch,
  :asech,
  :acoth,
]

for f in MATRIX_FUNCTIONS
  @eval begin
    function Base.$f(a::KroneckerArray)
      return throw(ArgumentError("Generic KroneckerArray `$($f)` is not supported."))
    end
  end
end

using LinearAlgebra: checksquare
function LinearAlgebra.det(a::KroneckerArray)
  checksquare(a.a)
  checksquare(a.b)
  return det(a.a) ^ size(a.b, 1) * det(a.b) ^ size(a.a, 1)
end

function LinearAlgebra.svd(a::KroneckerArray)
  Fa = svd(a.a)
  Fb = svd(a.b)
  return SVD(Fa.U ⊗ Fb.U, Fa.S ⊗ Fb.S, Fa.Vt ⊗ Fb.Vt)
end
function LinearAlgebra.svdvals(a::KroneckerArray)
  return svdvals(a.a) ⊗ svdvals(a.b)
end
function LinearAlgebra.eigen(a::KroneckerArray)
  Fa = eigen(a.a)
  Fb = eigen(a.b)
  return Eigen(Fa.values ⊗ Fb.values, Fa.vectors ⊗ Fb.vectors)
end
function LinearAlgebra.eigvals(a::KroneckerArray)
  return eigvals(a.a) ⊗ eigvals(a.b)
end

struct KroneckerQ{A,B}
  a::A
  b::B
end
function Base.:*(a::KroneckerQ, b::KroneckerQ)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.:*(a::KroneckerQ, b::KroneckerArray)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.:*(a::KroneckerArray, b::KroneckerQ)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.adjoint(a::KroneckerQ)
  return KroneckerQ(a.a', a.b')
end

struct KroneckerQR{QQ,RR}
  Q::QQ
  R::RR
end
Base.iterate(F::KroneckerQR) = (F.Q, Val(:R))
Base.iterate(F::KroneckerQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::KroneckerQR, ::Val{:done}) = nothing
function ⊗(a::LinearAlgebra.QRCompactWYQ, b::LinearAlgebra.QRCompactWYQ)
  return KroneckerQ(a, b)
end
function LinearAlgebra.qr(a::KroneckerArray)
  Fa = qr(a.a)
  Fb = qr(a.b)
  return KroneckerQR(Fa.Q ⊗ Fb.Q, Fa.R ⊗ Fb.R)
end

struct KroneckerLQ{LL,QQ}
  L::LL
  Q::QQ
end
Base.iterate(F::KroneckerLQ) = (F.L, Val(:Q))
Base.iterate(F::KroneckerLQ, ::Val{:Q}) = (F.Q, Val(:done))
Base.iterate(F::KroneckerLQ, ::Val{:done}) = nothing
function ⊗(a::LinearAlgebra.LQPackedQ, b::LinearAlgebra.LQPackedQ)
  return KroneckerQ(a, b)
end
function LinearAlgebra.lq(a::KroneckerArray)
  Fa = lq(a.a)
  Fb = lq(a.b)
  return KroneckerLQ(Fa.L ⊗ Fb.L, Fa.Q ⊗ Fb.Q)
end
