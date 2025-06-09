module KroneckerArrays

using GPUArraysCore: GPUArraysCore

export ⊗, ×

struct CartesianProduct{A,B}
  a::A
  b::B
end
arguments(a::CartesianProduct) = (a.a, a.b)
arguments(a::CartesianProduct, n::Int) = arguments(a)[n]

function Base.show(io::IO, a::CartesianProduct)
  print(io, a.a, " × ", a.b)
  return nothing
end

×(a, b) = CartesianProduct(a, b)
Base.length(a::CartesianProduct) = length(a.a) * length(a.b)
Base.getindex(a::CartesianProduct, i::CartesianProduct) = a.a[i.a] × a.b[i.b]

function Base.iterate(a::CartesianProduct, state...)
  x = iterate(Iterators.product(a.a, a.b), state...)
  isnothing(x) && return x
  next, new_state = x
  return ×(next...), new_state
end

struct CartesianProductUnitRange{T,P<:CartesianProduct,R<:AbstractUnitRange{T}} <:
       AbstractUnitRange{T}
  product::P
  range::R
end
Base.first(r::CartesianProductUnitRange) = first(r.range)
Base.last(r::CartesianProductUnitRange) = last(r.range)

cartesianproduct(r::CartesianProductUnitRange) = getfield(r, :product)
unproduct(r::CartesianProductUnitRange) = getfield(r, :range)

function CartesianProductUnitRange(p::CartesianProduct)
  return CartesianProductUnitRange(p, Base.OneTo(length(p)))
end
function CartesianProductUnitRange(a, b)
  return CartesianProductUnitRange(a × b)
end
to_range(a::AbstractUnitRange) = a
to_range(i::Integer) = Base.OneTo(i)
cartesianrange(a, b) = cartesianrange(to_range(a) × to_range(b))
function cartesianrange(p::CartesianProduct)
  p′ = to_range(p.a) × to_range(p.b)
  return cartesianrange(p′, Base.OneTo(length(p′)))
end
function cartesianrange(p::CartesianProduct, range::AbstractUnitRange)
  p′ = to_range(p.a) × to_range(p.b)
  return CartesianProductUnitRange(p′, range)
end

function Base.axes(r::CartesianProductUnitRange)
  return (CartesianProductUnitRange(r.product, only(axes(r.range))),)
end

using Base.Broadcast: DefaultArrayStyle
for f in (:+, :-)
  @eval begin
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), r::CartesianProductUnitRange, x::Integer
    )
      return CartesianProductUnitRange(r.product, $f.(r.range, x))
    end
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), x::Integer, r::CartesianProductUnitRange
    )
      return CartesianProductUnitRange(r.product, $f.(x, r.range))
    end
  end
end

struct KroneckerArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} <: AbstractArray{T,N}
  a::A
  b::B
end
function KroneckerArray(a::AbstractArray, b::AbstractArray)
  if ndims(a) != ndims(b)
    throw(
      ArgumentError("Kronecker product requires arrays of the same number of dimensions.")
    )
  end
  elt = promote_type(eltype(a), eltype(b))
  return KroneckerArray(convert(AbstractArray{elt}, a), convert(AbstractArray{elt}, b))
end
const KroneckerMatrix{T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}} = KroneckerArray{T,2,A,B}
const KroneckerVector{T,A<:AbstractVector{T},B<:AbstractVector{T}} = KroneckerArray{T,1,A,B}

function Base.copy(a::KroneckerArray)
  return copy(a.a) ⊗ copy(a.b)
end
function Base.copyto!(dest::KroneckerArray, src::KroneckerArray)
  copyto!(dest.a, src.a)
  copyto!(dest.b, src.b)
  return dest
end

function Base.similar(
  a::AbstractArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(a, elt, map(ax -> ax.product.a, axs)) ⊗
         similar(a, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  a::KroneckerArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(a.a, elt, map(ax -> ax.product.a, axs)) ⊗
         similar(a.b, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  arrayt::Type{<:AbstractArray},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(arrayt, map(ax -> ax.product.a, axs)) ⊗
         similar(arrayt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  arrayt::Type{<:KroneckerArray{<:Any,<:Any,A,B}},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
) where {A,B}
  return similar(A, map(ax -> ax.product.a, axs)) ⊗ similar(B, map(ax -> ax.product.b, axs))
end
function Base.similar(
  ::Type{<:KroneckerArray{<:Any,<:Any,A,B}}, sz::Tuple{Int,Vararg{Int}}
) where {A,B}
  return similar(promote_type(A, B), sz)
end

function flatten(t::Tuple{Tuple,Tuple,Vararg{Tuple}})
  return (t[1]..., flatten(Base.tail(t))...)
end
function flatten(t::Tuple{Tuple})
  return t[1]
end
flatten(::Tuple{}) = ()
function interleave(x::Tuple, y::Tuple)
  length(x) == length(y) || throw(ArgumentError("Tuples must have the same length."))
  xy = ntuple(i -> (x[i], y[i]), length(x))
  return flatten(xy)
end
# TODO: Maybe use scalar indexing based on KroneckerProducts.jl logic for cartesian indexing:
# https://github.com/perrutquist/KroneckerProducts.jl/blob/8c0104caf1f17729eb067259ba1473986121d032/src/KroneckerProducts.jl#L59-L66
function kron_nd(a::AbstractArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N}
  a′ = reshape(a, interleave(size(a), ntuple(one, N)))
  b′ = reshape(b, interleave(ntuple(one, N), size(b)))
  c′ = permutedims(a′ .* b′, reverse(ntuple(identity, 2N)))
  sz = ntuple(i -> size(a, i) * size(b, i), N)
  return permutedims(reshape(c′, sz), reverse(ntuple(identity, N)))
end
kron_nd(a::AbstractMatrix, b::AbstractMatrix) = kron(a, b)
kron_nd(a::AbstractVector, b::AbstractVector) = kron(a, b)

Base.collect(a::KroneckerArray) = kron_nd(a.a, a.b)

function Base.Array{T,N}(a::KroneckerArray{S,N}) where {T,S,N}
  return convert(Array{T,N}, collect(a))
end

Base.size(a::KroneckerArray) = ntuple(dim -> size(a.a, dim) * size(a.b, dim), ndims(a))

function Base.axes(a::KroneckerArray)
  return ntuple(ndims(a)) do dim
    return CartesianProductUnitRange(
      axes(a.a, dim) × axes(a.b, dim), Base.OneTo(size(a, dim))
    )
  end
end

arguments(a::KroneckerArray) = (a.a, a.b)
arguments(a::KroneckerArray, n::Int) = arguments(a)[n]
argument_types(a::KroneckerArray) = argument_types(typeof(a))
argument_types(::Type{<:KroneckerArray{<:Any,<:Any,A,B}}) where {A,B} = (A, B)

function Base.print_array(io::IO, a::KroneckerArray)
  Base.print_array(io, a.a)
  println(io, "\n ⊗")
  Base.print_array(io, a.b)
  return nothing
end
function Base.show(io::IO, a::KroneckerArray)
  show(io, a.a)
  print(io, " ⊗ ")
  show(io, a.b)
  return nothing
end

⊗(a::AbstractArray, b::AbstractArray) = KroneckerArray(a, b)
⊗(a::Number, b::Number) = a * b
⊗(a::Number, b::AbstractArray) = a * b
⊗(a::AbstractArray, b::Number) = a * b

function Base.getindex(a::KroneckerArray, i::Integer)
  return a[CartesianIndices(a)[i]]
end

# TODO: Use this logic from KroneckerProducts.jl for cartesian indexing
# in the n-dimensional case and use it to replace the matrix and vector cases:
# https://github.com/perrutquist/KroneckerProducts.jl/blob/8c0104caf1f17729eb067259ba1473986121d032/src/KroneckerProducts.jl#L59-L66
function Base.getindex(a::KroneckerArray{<:Any,N}, I::Vararg{Integer,N}) where {N}
  return error("Not implemented.")
end

function Base.getindex(a::KroneckerMatrix, i1::Integer, i2::Integer)
  GPUArraysCore.assertscalar("getindex")
  # Code logic from Kronecker.jl:
  # https://github.com/MichielStock/Kronecker.jl/blob/v0.5.5/src/base.jl#L101-L105
  k, l = size(a.b)
  return a.a[cld(i1, k), cld(i2, l)] * a.b[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

function Base.getindex(a::KroneckerVector, i::Integer)
  GPUArraysCore.assertscalar("getindex")
  k = length(a.b)
  return a.a[cld(i, k)] * a.b[(i - 1) % k + 1]
end

## function Base.getindex(a::KroneckerVector, i::CartesianProduct)
##   return a.a[i.a] ⊗ a.b[i.b]
## end
function Base.getindex(a::KroneckerArray{<:Any,N}, I::Vararg{CartesianProduct,N}) where {N}
  return a.a[map(Base.Fix2(getfield, :a), I)...] ⊗ a.b[map(Base.Fix2(getfield, :b), I)...]
end
# Fix ambigiuity error.
Base.getindex(a::KroneckerArray{<:Any,0}) = a.a[] * a.b[]

function Base.:(==)(a::KroneckerArray, b::KroneckerArray)
  return a.a == b.a && a.b == b.b
end
function Base.isapprox(a::KroneckerArray, b::KroneckerArray; kwargs...)
  return isapprox(a.a, b.a; kwargs...) && isapprox(a.b, b.b; kwargs...)
end
function Base.iszero(a::KroneckerArray)
  return iszero(a.a) || iszero(a.b)
end
function Base.isreal(a::KroneckerArray)
  return isreal(a.a) && isreal(a.b)
end
function Base.inv(a::KroneckerArray)
  return inv(a.a) ⊗ inv(a.b)
end
using LinearAlgebra: LinearAlgebra, pinv
function LinearAlgebra.pinv(a::KroneckerArray; kwargs...)
  return pinv(a.a; kwargs...) ⊗ pinv(a.b; kwargs...)
end
function Base.transpose(a::KroneckerArray)
  return transpose(a.a) ⊗ transpose(a.b)
end
function Base.adjoint(a::KroneckerArray)
  return a.a' ⊗ a.b'
end

function Base.:*(a::Number, b::KroneckerArray)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::KroneckerArray, b::Number)
  return a.a ⊗ (a.b * b)
end
function Base.:/(a::KroneckerArray, b::Number)
  return a * inv(b)
end

function Base.:-(a::KroneckerArray)
  return (-a.a) ⊗ a.b
end
for op in (:+, :-)
  @eval begin
    function Base.$op(a::KroneckerArray, b::KroneckerArray)
      if a.b == b.b
        return $op(a.a, b.a) ⊗ a.b
      elseif a.a == b.a
        return a.a ⊗ $op(a.b, b.b)
      end
      return throw(
        ArgumentError(
          "KroneckerArray addition is only supported when the first or secord arguments match.",
        ),
      )
    end
  end
end

using Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted
struct KroneckerStyle{N,A,B} <: AbstractArrayStyle{N} end
function KroneckerStyle{N}(a::BroadcastStyle, b::BroadcastStyle) where {N}
  return KroneckerStyle{N,a,b}()
end
function KroneckerStyle(a::AbstractArrayStyle{N}, b::AbstractArrayStyle{N}) where {N}
  return KroneckerStyle{N}(a, b)
end
function KroneckerStyle{N,A,B}(v::Val{M}) where {N,A,B,M}
  return KroneckerStyle{M,typeof(A)(v),typeof(B)(v)}()
end
function Base.BroadcastStyle(::Type{<:KroneckerArray{<:Any,N,A,B}}) where {N,A,B}
  return KroneckerStyle{N}(BroadcastStyle(A), BroadcastStyle(B))
end
function Base.BroadcastStyle(style1::KroneckerStyle{N}, style2::KroneckerStyle{N}) where {N}
  return KroneckerStyle{N}(
    BroadcastStyle(style1.a, style2.a), BroadcastStyle(style1.b, style2.b)
  )
end
function Base.similar(bc::Broadcasted{<:KroneckerStyle{N,A,B}}, elt::Type) where {N,A,B}
  ax_a = map(ax -> ax.product.a, axes(bc))
  ax_b = map(ax -> ax.product.b, axes(bc))
  bc_a = Broadcasted(A, nothing, (), ax_a)
  bc_b = Broadcasted(B, nothing, (), ax_b)
  a = similar(bc_a, elt)
  b = similar(bc_b, elt)
  return a ⊗ b
end
function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:KroneckerStyle})
  return throw(
    ArgumentError(
      "Arbitrary broadcasting is not supported for KroneckerArrays since they might not preserve the Kronecker structure.",
    ),
  )
end

function Base.map(f, a1::KroneckerArray, a_rest::KroneckerArray...)
  return throw(
    ArgumentError(
      "Arbitrary mapping is not supported for KroneckerArrays since they might not preserve the Kronecker structure.",
    ),
  )
end
function Base.map!(f, dest::KroneckerArray, a1::KroneckerArray, a_rest::KroneckerArray...)
  return throw(
    ArgumentError(
      "Arbitrary mapping is not supported for KroneckerArrays since they might not preserve the Kronecker structure.",
    ),
  )
end
function Base.map!(::typeof(identity), dest::KroneckerArray, a::KroneckerArray)
  dest.a .= a.a
  dest.b .= a.b
  return dest
end
for f in [:+, :-]
  @eval begin
    function Base.map!(
      ::typeof($f), dest::KroneckerArray, a::KroneckerArray, b::KroneckerArray
    )
      if a.b == b.b
        map!($f, dest.a, a.a, b.a)
        dest.b .= a.b
      elseif a.a == b.a
        dest.a .= a.a
        map!($f, dest.b, a.b, b.b)
      else
        throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or second arguments match.",
          ),
        )
      end
      return dest
    end
  end
end
function Base.map!(
  f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerArray, a::KroneckerArray
)
  dest.a .= f.f.(f.x, a.a)
  dest.b .= a.b
  return dest
end
function Base.map!(
  f::Base.Fix2{typeof(*),<:Number}, dest::KroneckerArray, a::KroneckerArray
)
  dest.a .= a.a
  dest.b .= f.f.(a.b, f.x)
  return dest
end
function Base.map!(
  f::Base.Fix2{typeof(/),<:Number}, dest::KroneckerArray, a::KroneckerArray
)
  return map!(Base.Fix2(*, inv(f.x)), dest, a)
end
function Base.map!(::typeof(conj), dest::KroneckerArray, a::KroneckerArray)
  dest.a .= conj.(a.a)
  dest.b .= conj.(a.b)
  return dest
end

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

using DiagonalArrays: DiagonalArrays, diagonal
function DiagonalArrays.diagonal(a::KroneckerArray)
  return diagonal(a.a) ⊗ diagonal(a.b)
end

function Base.:*(a::KroneckerArray, b::KroneckerArray)
  return (a.a * b.a) ⊗ (a.b * b.b)
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

function Base.real(a::KroneckerArray)
  if iszero(imag(a.a)) || iszero(imag(a.b))
    return real(a.a) ⊗ real(a.b)
  elseif iszero(real(a.a)) || iszero(real(a.b))
    return -imag(a.a) ⊗ imag(a.b)
  end
  return real(a.a) ⊗ real(a.b) - imag(a.a) ⊗ imag(a.b)
end
function Base.imag(a::KroneckerArray)
  if iszero(imag(a.a)) || iszero(real(a.b))
    return real(a.a) ⊗ imag(a.b)
  elseif iszero(real(a.a)) || iszero(imag(a.b))
    return imag(a.a) ⊗ real(a.b)
  end
  return real(a.a) ⊗ imag(a.b) + imag(a.a) ⊗ real(a.b)
end

using MatrixAlgebraKit: MatrixAlgebraKit, diagview
function MatrixAlgebraKit.diagview(a::KroneckerMatrix)
  return diagview(a.a) ⊗ diagview(a.b)
end
function LinearAlgebra.diag(a::KroneckerArray)
  return copy(diagview(a.a)) ⊗ copy(diagview(a.b))
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

using DerivableInterfaces: DerivableInterfaces, zero!
function DerivableInterfaces.zero!(a::KroneckerArray)
  zero!(a.a)
  zero!(a.b)
  return a
end

using FillArrays: Eye
const EyeKronecker{T,A<:Eye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerEye{T,A<:AbstractMatrix{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}
const EyeEye{T,A<:Eye{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}

using DerivableInterfaces: DerivableInterfaces, zero!
function DerivableInterfaces.zero!(a::EyeKronecker)
  zero!(a.b)
  return a
end
function DerivableInterfaces.zero!(a::KroneckerEye)
  zero!(a.a)
  return a
end
function DerivableInterfaces.zero!(a::EyeEye)
  return throw(ArgumentError("Can't zero out `Eye ⊗ Eye`."))
end

function Base.:*(a::Number, b::EyeKronecker)
  return b.a ⊗ (a * b.b)
end
function Base.:*(a::Number, b::KroneckerEye)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::Number, b::EyeEye)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::EyeKronecker, b::Number)
  return a.a ⊗ (a.b * b)
end
function Base.:*(a::KroneckerEye, b::Number)
  return (a.a * b) ⊗ a.b
end
function Base.:*(a::EyeEye, b::Number)
  return a.a ⊗ (a.b * b)
end

function Base.:-(a::EyeKronecker)
  return a.a ⊗ (-a.b)
end
function Base.:-(a::KroneckerEye)
  return (-a.a) ⊗ a.b
end
function Base.:-(a::EyeEye)
  return (-a.a) ⊗ a.b
end
for op in (:+, :-)
  @eval begin
    function Base.$op(a::EyeKronecker, b::EyeKronecker)
      if a.a ≠ b.a
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return a.a ⊗ $op(a.b, b.b)
    end
    function Base.$op(a::KroneckerEye, b::KroneckerEye)
      if a.b ≠ b.b
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return $op(a.a, b.a) ⊗ a.b
    end
    function Base.$op(a::EyeEye, b::EyeEye)
      if a.b ≠ b.b
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return $op(a.a, b.a) ⊗ a.b
    end
  end
end

function Base.map!(::typeof(identity), dest::EyeKronecker, a::EyeKronecker)
  dest.b .= a.b
  return dest
end
function Base.map!(::typeof(identity), dest::KroneckerEye, a::KroneckerEye)
  dest.a .= a.a
  return dest
end
function Base.map!(::typeof(identity), dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
for f in [:+, :-]
  @eval begin
    function Base.map!(::typeof($f), dest::EyeKronecker, a::EyeKronecker, b::EyeKronecker)
      if dest.a ≠ a.a ≠ b.a
        throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or second arguments match.",
          ),
        )
      end
      map!($f, dest.b, a.b, b.b)
      return dest
    end
    function Base.map!(::typeof($f), dest::KroneckerEye, a::KroneckerEye, b::KroneckerEye)
      if dest.b ≠ a.b ≠ b.b
        throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or second arguments match.",
          ),
        )
      end
      map!($f, dest.a, a.a, b.a)
      return dest
    end
    function Base.map!(::typeof($f), dest::EyeEye, a::EyeEye, b::EyeEye)
      return error("Can't write in-place.")
    end
  end
end
function Base.map!(f::typeof(-), dest::EyeKronecker, a::EyeKronecker)
  map!(f, dest.b, a.b)
  return dest
end
function Base.map!(f::typeof(-), dest::KroneckerEye, a::KroneckerEye)
  map!(f, dest.a, a.a)
  return dest
end
function Base.map!(f::typeof(-), dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::EyeKronecker, a::EyeKronecker)
  dest.b .= f.f.(f.x, a.b)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  dest.a .= f.f.(f.x, a.a)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeKronecker, a::EyeKronecker)
  dest.b .= f.f.(a.b, f.x)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  dest.a .= f.f.(a.a, f.x)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end

using MatrixAlgebraKit:
  MatrixAlgebraKit,
  AbstractAlgorithm,
  TruncationStrategy,
  default_eig_algorithm,
  default_eigh_algorithm,
  default_lq_algorithm,
  default_polar_algorithm,
  default_qr_algorithm,
  default_svd_algorithm,
  eig_full!,
  eig_trunc!,
  eig_vals!,
  eigh_full!,
  eigh_trunc!,
  eigh_vals!,
  initialize_output,
  left_null!,
  left_orth!,
  left_polar!,
  lq_compact!,
  lq_full!,
  qr_compact!,
  qr_full!,
  right_null!,
  right_orth!,
  right_polar!,
  svd_compact!,
  svd_full!,
  svd_trunc!,
  svd_vals!,
  truncate!

struct KroneckerAlgorithm{A,B} <: AbstractAlgorithm
  a::A
  b::B
end

using MatrixAlgebraKit:
  copy_input,
  eig_full,
  eig_vals,
  eigh_full,
  eigh_vals,
  qr_compact,
  qr_full,
  left_null,
  left_orth,
  left_polar,
  lq_compact,
  lq_full,
  right_null,
  right_orth,
  right_polar,
  svd_compact,
  svd_full

for f in [
  :eig_full,
  :eigh_full,
  :qr_compact,
  :qr_full,
  :left_polar,
  :lq_compact,
  :lq_full,
  :right_polar,
  :svd_compact,
  :svd_full,
]
  @eval begin
    function MatrixAlgebraKit.copy_input(::typeof($f), a::KroneckerMatrix)
      return copy_input($f, a.a) ⊗ copy_input($f, a.b)
    end
  end
end

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  @eval begin
    function MatrixAlgebraKit.$f(
      A::Type{<:KroneckerMatrix}; kwargs1=(;), kwargs2=(;), kwargs...
    )
      A1, A2 = argument_types(A)
      return KroneckerAlgorithm(
        $f(A1; kwargs..., kwargs1...), $f(A2; kwargs..., kwargs2...)
      )
    end
  end
end

# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(qr_compact!), A::Type{<:KroneckerMatrix}; kwargs...
)
  return default_qr_algorithm(A; kwargs...)
end
# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(qr_full!), A::Type{<:KroneckerMatrix}; kwargs...
)
  return default_qr_algorithm(A; kwargs...)
end

for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), a::KroneckerMatrix, alg::KroneckerAlgorithm
    )
      return initialize_output($f, a.a, alg.a) .⊗ initialize_output($f, a.b, alg.b)
    end
    function MatrixAlgebraKit.$f(
      a::KroneckerMatrix, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
    )
      $f(a.a, Base.Fix2(getfield, :a).(F), alg.a; kwargs..., kwargs1...)
      $f(a.b, Base.Fix2(getfield, :b).(F), alg.b; kwargs..., kwargs2...)
      return F
    end
  end
end

for f in [:eig_vals!, :eigh_vals!, :svd_vals!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), a::KroneckerMatrix, alg::KroneckerAlgorithm
    )
      return initialize_output($f, a.a, alg.a) ⊗ initialize_output($f, a.b, alg.b)
    end
    function MatrixAlgebraKit.$f(a::KroneckerMatrix, F, alg::KroneckerAlgorithm)
      $f(a.a, F.a, alg.a)
      $f(a.b, F.b, alg.b)
      return F
    end
  end
end

for f in [:left_orth!, :right_orth!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
      return initialize_output($f, a.a) .⊗ initialize_output($f, a.b)
    end
  end
end

for f in [:left_null!, :right_null!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
      return initialize_output($f, a.a) ⊗ initialize_output($f, a.b)
    end
    function MatrixAlgebraKit.$f(a::KroneckerMatrix, F; kwargs1=(;), kwargs2=(;), kwargs...)
      $f(a.a, F.a; kwargs..., kwargs1...)
      $f(a.b, F.b; kwargs..., kwargs2...)
      return F
    end
  end
end

####################################################################################
# Special cases for MatrixAlgebraKit factorizations of `Eye(n) ⊗ A` and
# `A ⊗ Eye(n)` where `A`.
# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/34
# is merged.

using FillArrays: SquareEye
const SquareEyeKronecker{T,A<:SquareEye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerSquareEye{T,A<:AbstractMatrix{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}
const SquareEyeSquareEye{T,A<:SquareEye{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}

# Special case of similar for `SquareEye ⊗ A` and `A ⊗ SquareEye`.
function Base.similar(
  a::SquareEyeKronecker,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_a = (only(unique(ax_a)),)
  return Eye{elt}(eye_ax_a) ⊗ similar(a.b, elt, ax_b)
end
function Base.similar(
  a::KroneckerSquareEye,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_b = (only(unique(ax_b)),)
  return similar(a.a, elt, ax_a) ⊗ Eye{elt}(eye_ax_b)
end
function Base.similar(
  a::SquareEyeSquareEye,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_a = (only(unique(ax_a)),)
  eye_ax_b = (only(unique(ax_b)),)
  return Eye{elt}(eye_ax_a) ⊗ Eye{elt}(eye_ax_b)
end

function Base.similar(
  arrayt::Type{<:SquareEyeKronecker{<:Any,<:Any,A}},
  axs::NTuple{2,CartesianProductUnitRange{<:Integer}},
) where {A}
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_a = (only(unique(ax_a)),)
  return Eye{eltype(arrayt)}(eye_ax_a) ⊗ similar(A, ax_b)
end
function Base.similar(
  arrayt::Type{<:KroneckerSquareEye{<:Any,A}},
  axs::NTuple{2,CartesianProductUnitRange{<:Integer}},
) where {A}
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_b = (only(unique(ax_b)),)
  return similar(A, ax_a) ⊗ Eye{eltype(arrayt)}(eye_ax_b)
end
function Base.similar(
  arrayt::Type{<:SquareEyeSquareEye}, axs::NTuple{2,CartesianProductUnitRange{<:Integer}}
)
  elt = eltype(arrayt)
  ax_a = map(ax -> ax.product.a, axs)
  ax_b = map(ax -> ax.product.b, axs)
  eye_ax_a = (only(unique(ax_a)),)
  eye_ax_b = (only(unique(ax_b)),)
  return Eye{elt}(eye_ax_a) ⊗ Eye{elt}(eye_ax_b)
end

struct SquareEyeAlgorithm{KWargs<:NamedTuple} <: AbstractAlgorithm
  kwargs::KWargs
end
SquareEyeAlgorithm(; kwargs...) = SquareEyeAlgorithm((; kwargs...))

# Defined to avoid type piracy.
_copy_input_squareeye(f::F, a) where {F} = copy_input(f, a)
_copy_input_squareeye(f::F, a::SquareEye) where {F} = a

for f in [
  :eig_full,
  :eig_vals,
  :eigh_full,
  :eigh_vals,
  :qr_compact,
  :qr_full,
  :left_null,
  :left_orth,
  :left_polar,
  :lq_compact,
  :lq_full,
  :right_null,
  :right_orth,
  :right_polar,
  :svd_compact,
  :svd_full,
]
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.copy_input(::typeof($f), a::$T)
        return _copy_input_squareeye($f, a.a) ⊗ _copy_input_squareeye($f, a.b)
      end
    end
  end
end

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a; kwargs...) = $f(a; kwargs...)
    $f′(a::Type{<:SquareEye}; kwargs...) = SquareEyeAlgorithm(; kwargs...)
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.$f(A::Type{<:$T}; kwargs1=(;), kwargs2=(;), kwargs...)
        A1, A2 = argument_types(A)
        return KroneckerAlgorithm(
          $f′(A1; kwargs..., kwargs1...), $f′(A2; kwargs..., kwargs2...)
        )
      end
    end
  end
end

# Defined to avoid type piracy.
_initialize_output_squareeye(f::F, a) where {F} = initialize_output(f, a)
_initialize_output_squareeye(f::F, a, alg) where {F} = initialize_output(f, a, alg)

for f in [:left_null!, :right_null!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = a
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = a
  end
end
for f in [
  :qr_compact!,
  :qr_full!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_orth!,
  :right_polar!,
]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = (a, a)
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, a)
  end
end
_initialize_output_squareeye(::typeof(eig_full!), a::SquareEye) = complex.((a, a))
_initialize_output_squareeye(::typeof(eig_full!), a::SquareEye, alg) = complex.((a, a))
_initialize_output_squareeye(::typeof(eigh_full!), a::SquareEye) = (real(a), a)
_initialize_output_squareeye(::typeof(eigh_full!), a::SquareEye, alg) = (real(a), a)
for f in [:svd_compact!, :svd_full!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = (a, real(a), a)
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, real(a), a)
  end
end

for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_orth!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F, alg; kwargs...) = $f(a, F, alg; kwargs...)
    $f′(a, F, alg::SquareEyeAlgorithm) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(::typeof($f), a::$T)
        return _initialize_output_squareeye($f, a.a) .⊗
               _initialize_output_squareeye($f, a.b)
      end
      function MatrixAlgebraKit.initialize_output(
        ::typeof($f), a::$T, alg::KroneckerAlgorithm
      )
        return _initialize_output_squareeye($f, a.a, alg.a) .⊗
               _initialize_output_squareeye($f, a.b, alg.b)
      end
      function MatrixAlgebraKit.$f(
        a::$T, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
      )
        $f′(a.a, Base.Fix2(getfield, :a).(F), alg.a; kwargs..., kwargs1...)
        $f′(a.b, Base.Fix2(getfield, :b).(F), alg.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

for f in [:left_null!, :right_null!]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F; kwargs...) = $f(a, F; kwargs...)
    $f′(a::SquareEye, F) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(::typeof($f), a::$T)
        return _initialize_output_squareeye($f, a.a) ⊗ _initialize_output_squareeye($f, a.b)
      end
      function MatrixAlgebraKit.$f(a::$T, F; kwargs1=(;), kwargs2=(;), kwargs...)
        $f′(a.a, F.a; kwargs..., kwargs1...)
        $f′(a.b, F.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

function MatrixAlgebraKit.initialize_output(f::typeof(left_null!), a::SquareEyeSquareEye)
  return _initialize_output_squareeye(f, a.a) ⊗ _initialize_output_squareeye(f, a.b)
end
function MatrixAlgebraKit.left_null!(
  a::SquareEyeSquareEye, F; kwargs1=(;), kwargs2=(;), kwargs...
)
  return throw(MethodError(left_null!, (a, F)))
end

function MatrixAlgebraKit.initialize_output(f::typeof(right_null!), a::SquareEyeSquareEye)
  return _initialize_output_squareeye(f, a.a) ⊗ _initialize_output_squareeye(f, a.b)
end
function MatrixAlgebraKit.right_null!(
  a::SquareEyeSquareEye, F; kwargs1=(;), kwargs2=(;), kwargs...
)
  return throw(MethodError(right_null!, (a, F)))
end

_initialize_output_squareeye(::typeof(eig_vals!), a::SquareEye) = parent(a)
_initialize_output_squareeye(::typeof(eig_vals!), a::SquareEye, alg) = parent(a)
for f in [:eigh_vals!, svd_vals!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = real(parent(a))
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = real(parent(a))
  end
end

for f in [:eig_vals!, :eigh_vals!, :svd_vals!]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F, alg; kwargs...) = $f(a, F, alg; kwargs...)
    $f′(a, F, alg::SquareEyeAlgorithm) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(
        ::typeof($f), a::$T, alg::KroneckerAlgorithm
      )
        return _initialize_output_squareeye($f, a.a, alg.a) ⊗
               _initialize_output_squareeye($f, a.b, alg.b)
      end
      function MatrixAlgebraKit.$f(
        a::$T, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
      )
        $f′(a.a, F.a, alg.a; kwargs..., kwargs1...)
        $f′(a.b, F.b, alg.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

using MatrixAlgebraKit: TruncationStrategy, diagview, findtruncated, truncate!

struct KroneckerTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

# Avoid instantiating the identity.
function Base.getindex(a::SquareEyeKronecker, I::Vararg{CartesianProduct{Colon},2})
  return a.a ⊗ a.b[I[1].b, I[2].b]
end
function Base.getindex(a::KroneckerSquareEye, I::Vararg{CartesianProduct{<:Any,Colon},2})
  return a.a[I[1].a, I[2].a] ⊗ a.b
end
function Base.getindex(a::SquareEyeSquareEye, I::Vararg{CartesianProduct{Colon,Colon},2})
  return a
end

using FillArrays: OnesVector
const OnesKroneckerVector{T,A<:OnesVector{T},B<:AbstractVector{T}} = KroneckerVector{T,A,B}
const KroneckerOnesVector{T,A<:AbstractVector{T},B<:OnesVector{T}} = KroneckerVector{T,A,B}
const OnesVectorOnesVector{T,A<:OnesVector{T},B<:OnesVector{T}} = KroneckerVector{T,A,B}

function MatrixAlgebraKit.findtruncated(
  values::OnesKroneckerVector, strategy::KroneckerTruncationStrategy
)
  I = findtruncated(Vector(values), strategy.strategy)
  prods = collect(only(axes(values)).product)[I]
  I_data = unique(map(x -> x.a, prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> x.a == i, prods) == length(values.a)
  end
  return (:) × I_data
end
function MatrixAlgebraKit.findtruncated(
  values::KroneckerOnesVector, strategy::KroneckerTruncationStrategy
)
  I = findtruncated(Vector(values), strategy.strategy)
  prods = collect(only(axes(values)).product)[I]
  I_data = unique(map(x -> x.b, prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> x.b == i, prods) == length(values.b)
  end
  return I_data × (:)
end
function MatrixAlgebraKit.findtruncated(
  values::OnesVectorOnesVector, strategy::KroneckerTruncationStrategy
)
  return throw(ArgumentError("Can't truncate Eye ⊗ Eye."))
end

for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f), DV::NTuple{2,KroneckerMatrix}, strategy::TruncationStrategy
    )
      return truncate!($f, DV, KroneckerTruncationStrategy(strategy))
    end
    function MatrixAlgebraKit.truncate!(
      ::typeof($f), (D, V)::NTuple{2,KroneckerMatrix}, strategy::KroneckerTruncationStrategy
    )
      I = findtruncated(diagview(D), strategy)
      return (D[I, I], V[(:) × (:), I])
    end
  end
end

function MatrixAlgebraKit.truncate!(
  f::typeof(svd_trunc!), USVᴴ::NTuple{3,KroneckerMatrix}, strategy::TruncationStrategy
)
  return truncate!(f, USVᴴ, KroneckerTruncationStrategy(strategy))
end
function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::NTuple{3,KroneckerMatrix},
  strategy::KroneckerTruncationStrategy,
)
  I = findtruncated(diagview(S), strategy)
  return (U[(:) × (:), I], S[I, I], Vᴴ[I, (:) × (:)])
end

for f in MATRIX_FUNCTIONS
  @eval begin
    function Base.$f(a::SquareEyeKronecker)
      return a.a ⊗ $f(a.b)
    end
    function Base.$f(a::KroneckerSquareEye)
      return $f(a.a) ⊗ a.b
    end
    function Base.$f(a::SquareEyeSquareEye)
      return throw(ArgumentError("`$($f)` on `Eye ⊗ Eye` is not supported."))
    end
  end
end

function LinearAlgebra.pinv(a::SquareEyeKronecker; kwargs...)
  return a.a ⊗ pinv(a.b; kwargs...)
end
function LinearAlgebra.pinv(a::KroneckerSquareEye; kwargs...)
  return pinv(a.a; kwargs...) ⊗ a.b
end
function LinearAlgebra.pinv(a::SquareEyeSquareEye; kwargs...)
  return a
end

end
