# Allows customizing for `FillArrays.Eye`.
function _convert(A::Type{<:AbstractArray}, a::AbstractArray)
  return convert(A, a)
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
  return KroneckerArray(_convert(AbstractArray{elt}, a), _convert(AbstractArray{elt}, b))
end
const KroneckerMatrix{T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}} = KroneckerArray{T,2,A,B}
const KroneckerVector{T,A<:AbstractVector{T},B<:AbstractVector{T}} = KroneckerArray{T,1,A,B}

arg1(a::KroneckerArray) = a.a
arg2(a::KroneckerArray) = a.b

using Adapt: Adapt, adapt
_adapt(to, a::AbstractArray) = adapt(to, a)
Adapt.adapt_structure(to, a::KroneckerArray) = _adapt(to, a.a) ⊗ _adapt(to, a.b)

# Allows extra customization, like for `FillArrays.Eye`.
_copy(a::AbstractArray) = copy(a)

function Base.copy(a::KroneckerArray)
  return _copy(a.a) ⊗ _copy(a.b)
end
function Base.copyto!(dest::KroneckerArray, src::KroneckerArray)
  copyto!(dest.a, src.a)
  copyto!(dest.b, src.b)
  return dest
end

# Like `similar` but allows some custom behavior, such as for `FillArrays.Eye`.
function _similar(a::AbstractArray, elt::Type, axs::Tuple{Vararg{AbstractUnitRange}})
  return similar(a, elt, axs)
end
function _similar(arrayt::Type{<:AbstractArray}, axs::Tuple{Vararg{AbstractUnitRange}})
  return similar(arrayt, axs)
end

function Base.similar(
  a::AbstractArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return _similar(a, elt, map(ax -> ax.product.a, axs)) ⊗
         _similar(a, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  a::KroneckerArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return _similar(a.a, elt, map(ax -> ax.product.a, axs)) ⊗
         _similar(a.b, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  arrayt::Type{<:AbstractArray},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return _similar(arrayt, map(ax -> ax.product.a, axs)) ⊗
         _similar(arrayt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  arrayt::Type{<:KroneckerArray{<:Any,<:Any,A,B}},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
) where {A,B}
  return _similar(A, map(ax -> ax.product.a, axs)) ⊗
         _similar(B, map(ax -> ax.product.b, axs))
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

# Eagerly collect arguments to make more general on GPU.
Base.collect(a::KroneckerArray) = kron_nd(collect(a.a), collect(a.b))

Base.zero(a::KroneckerArray) = zero(arg1(a)) ⊗ zero(arg2(a))

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

using GPUArraysCore: GPUArraysCore
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

using DiagonalArrays: DiagonalArrays, diagonal
function DiagonalArrays.diagonal(a::KroneckerArray)
  return diagonal(a.a) ⊗ diagonal(a.b)
end

Base.real(a::KroneckerArray{<:Real}) = a
function Base.real(a::KroneckerArray)
  if iszero(imag(a.a)) || iszero(imag(a.b))
    return real(a.a) ⊗ real(a.b)
  elseif iszero(real(a.a)) || iszero(real(a.b))
    return -imag(a.a) ⊗ imag(a.b)
  end
  return real(a.a) ⊗ real(a.b) - imag(a.a) ⊗ imag(a.b)
end
Base.imag(a::KroneckerArray{<:Real}) = zero(a)
function Base.imag(a::KroneckerArray)
  if iszero(imag(a.a)) || iszero(real(a.b))
    return real(a.a) ⊗ imag(a.b)
  elseif iszero(real(a.a)) || iszero(imag(a.b))
    return imag(a.a) ⊗ real(a.b)
  end
  return real(a.a) ⊗ imag(a.b) + imag(a.a) ⊗ real(a.b)
end

for f in [:transpose, :adjoint, :inv]
  @eval begin
    function Base.$f(a::KroneckerArray)
      return $f(a.a) ⊗ $f(a.b)
    end
  end
end

# Allows for customizations for FillArrays.
_BroadcastStyle(x) = BroadcastStyle(x)

using Base.Broadcast: Broadcast, AbstractArrayStyle, BroadcastStyle, Broadcasted
struct KroneckerStyle{N,A,B} <: AbstractArrayStyle{N} end
arg1(::Type{<:KroneckerStyle{<:Any,A}}) where {A} = A
arg1(style::KroneckerStyle) = arg1(typeof(style))
arg2(::Type{<:KroneckerStyle{<:Any,B}}) where {B} = B
arg2(style::KroneckerStyle) = arg2(typeof(style))
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
  return KroneckerStyle{N}(_BroadcastStyle(A), _BroadcastStyle(B))
end
function Base.BroadcastStyle(style1::KroneckerStyle{N}, style2::KroneckerStyle{N}) where {N}
  style_a = BroadcastStyle(arg1(style1), arg1(style2))
  (style_a isa Broadcast.Unknown) && return Broadcast.Unknown()
  style_b = BroadcastStyle(arg2(style1), arg2(style2))
  (style_b isa Broadcast.Unknown) && return Broadcast.Unknown()
  return KroneckerStyle{N}(style_a, style_b)
end
function Base.similar(bc::Broadcasted{<:KroneckerStyle{N,A,B}}, elt::Type, ax) where {N,A,B}
  ax_a = arg1.(ax)
  ax_b = arg2.(ax)
  bc_a = Broadcasted(A, nothing, (), ax_a)
  bc_b = Broadcasted(B, nothing, (), ax_b)
  a = similar(bc_a, elt)
  b = similar(bc_b, elt)
  return a ⊗ b
end

function Base.map(f, a1::KroneckerArray, a_rest::KroneckerArray...)
  return Broadcast.broadcast_preserving_zero_d(f, a1, a_rest...)
end
function Base.map!(f, dest::KroneckerArray, a1::KroneckerArray, a_rest::KroneckerArray...)
  dest .= f.(a1, a_rest...)
  return dest
end

function Base.copyto!(dest::KroneckerArray, a::Sum{<:KroneckerStyle})
  dest1 = arg1(dest)
  dest2 = arg2(dest)
  f = LinearCombination(a)
  args = arguments(a)
  arg1s = arg1.(args)
  arg2s = arg2.(args)
  if allequal(arg2s)
    copyto!(dest2, first(arg2s))
    dest1 .= f.(arg1s...)
  elseif allequal(arg1s)
    copyto!(dest1, first(arg1s))
    dest2 .= f.(arg2s...)
  else
    error("This operation doesn't preserve the Kronecker structure.")
  end
  return dest
end

function Broadcast.broadcasted(::KroneckerStyle, f, as...)
  return error("Arbitrary broadcasting not supported for KroneckerArray.")
end

# Linear operations.
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(+), a, b)
  return Sum(a) + Sum(b)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(-), a, b)
  return Sum(a) - Sum(b)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), c::Number, a)
  return c * Sum(a)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), a, c::Number)
  return Sum(a) * c
end
# Fix ambiguity error.
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), a::Number, b::Number)
  return a * b
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(/), a, c::Number)
  return Sum(a) / c
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(-), a)
  return -Sum(a)
end

# Rewrite rules to canonicalize broadcast expressions.
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix1{typeof(*),<:Number}, a)
  return broadcasted(style, *, f.x, a)
end
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(*),<:Number}, a)
  return broadcasted(style, *, a, f.x)
end
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(/),<:Number}, a)
  return broadcasted(style, /, a, f.x)
end

# Use to determine the element type of KroneckerBroadcasted.
_eltype(x) = eltype(x)
_eltype(x::Broadcasted) = Base.promote_op(x.f, _eltype.(x.args)...)

using Base.Broadcast: broadcasted
struct KroneckerBroadcasted{A<:Broadcasted,B<:Broadcasted}
  a::A
  b::B
end
arg1(a::KroneckerBroadcasted) = a.a
arg2(a::KroneckerBroadcasted) = a.b
⊗(a::Broadcasted, b::Broadcasted) = KroneckerBroadcasted(a, b)
Broadcast.materialize(a::KroneckerBroadcasted) = copy(a)
Broadcast.materialize!(dest, a::KroneckerBroadcasted) = copyto!(dest, a)
Broadcast.broadcastable(a::KroneckerBroadcasted) = a
Base.copy(a::KroneckerBroadcasted) = copy(arg1(a)) ⊗ copy(arg2(a))
function Base.copyto!(dest::KroneckerArray, a::KroneckerBroadcasted)
  copyto!(arg1(dest), copy(arg1(a)))
  copyto!(arg2(dest), copy(arg2(a)))
  return dest
end
function Base.eltype(a::KroneckerBroadcasted)
  a1 = arg1(a)
  a2 = arg2(a)
  return Base.promote_op(*, _eltype(a1), _eltype(a2))
end
function Base.axes(a::KroneckerBroadcasted)
  ax1 = axes(arg1(a))
  ax2 = axes(arg2(a))
  return cartesianrange.(ax1 .× ax2)
end

function Base.BroadcastStyle(
  ::Type{<:KroneckerBroadcasted{A,B}}
) where {StyleA,StyleB,A<:Broadcasted{StyleA},B<:Broadcasted{StyleB}}
  @assert ndims(A) == ndims(B)
  N = ndims(A)
  return KroneckerStyle{N}(StyleA(), StyleB())
end

# Operations that preserve the Kronecker structure.
for f in [:identity, :conj]
  @eval begin
    function Broadcast.broadcasted(::KroneckerStyle{<:Any,A,B}, ::typeof($f), a) where {A,B}
      return broadcasted(A, $f, arg1(a)) ⊗ broadcasted(B, $f, arg2(a))
    end
  end
end

## using MapBroadcast: MapBroadcast, MapFunction
## function Base.broadcasted(
##   style::KroneckerStyle,
##   f::MapFunction{typeof(*),<:Tuple{<:Number,MapBroadcast.Arg}},
##   a::KroneckerArray,
## )
##   return broadcasted(style, Base.Fix1(*, f.args[1]), a)
## end
## function Base.broadcasted(
##   style::KroneckerStyle,
##   f::MapFunction{typeof(*),<:Tuple{MapBroadcast.Arg,<:Number}},
##   a::KroneckerArray,
## )
##   return broadcasted(style, Base.Fix2(*, f.args[2]), a)
## end
## function Base.broadcasted(
##   style::KroneckerStyle,
##   f::MapFunction{typeof(/),<:Tuple{MapBroadcast.Arg,<:Number}},
##   a::KroneckerArray,
## )
##   return broadcasted(style, Base.Fix2(/, f.args[2]), a)
## end
