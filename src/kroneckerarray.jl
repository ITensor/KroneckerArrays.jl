# Allows customizing for `FillArrays.Eye`.
function _convert(A::Type{<:AbstractArray}, a::AbstractArray)
  return convert(A, a)
end
# Custom `_convert` works around the issue that
# `convert(::Type{<:Diagonal}, ::AbstractMatrix)` isnt' defined
# in Julia v1.10 (https://github.com/JuliaLang/julia/pull/48895,
# https://github.com/JuliaLang/julia/pull/52487).
# TODO: Delete once we drop support for Julia v1.10.
using LinearAlgebra: LinearAlgebra, Diagonal, diag, isdiag
_construct(A::Type{<:Diagonal}, a::AbstractMatrix) = A(diag(a))
function _convert(A::Type{<:Diagonal}, a::AbstractMatrix)
  LinearAlgebra.checksquare(a)
  return isdiag(a) ? _construct(A, a) : throw(InexactError(:convert, A, a))
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
Adapt.adapt_structure(to, a::KroneckerArray) = _adapt(to, arg1(a)) ⊗ _adapt(to, arg2(a))

# Allows extra customization, like for `FillArrays.Eye`.
_copy(a::AbstractArray) = copy(a)

function Base.copy(a::KroneckerArray)
  return _copy(arg1(a)) ⊗ _copy(arg2(a))
end
function Base.copyto!(dest::KroneckerArray, src::KroneckerArray)
  copyto!(arg1(dest), arg1(src))
  copyto!(arg2(dest), arg2(src))
  return dest
end

function Base.convert(::Type{KroneckerArray{T,N,A,B}}, a::KroneckerArray) where {T,N,A,B}
  return KroneckerArray(_convert(A, arg1(a)), _convert(B, arg2(a)))
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
  return _similar(a, elt, map(arg1, axs)) ⊗ _similar(a, elt, map(arg2, axs))
end
function Base.similar(
  a::KroneckerArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return _similar(arg1(a), elt, map(arg1, axs)) ⊗ _similar(arg2(a), elt, map(arg2, axs))
end
function Base.similar(
  arrayt::Type{<:AbstractArray},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return _similar(arrayt, map(arg1, axs)) ⊗ _similar(arrayt, map(arg2, axs))
end
function Base.similar(
  arrayt::Type{<:KroneckerArray{<:Any,<:Any,A,B}},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
) where {A,B}
  return _similar(A, map(arg1, axs)) ⊗ _similar(B, map(arg2, axs))
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
Base.collect(a::KroneckerArray) = kron_nd(collect(arg1(a)), collect(arg2(a)))

Base.zero(a::KroneckerArray) = zero(arg1(a)) ⊗ zero(arg2(a))

function Base.Array{T,N}(a::KroneckerArray{S,N}) where {T,S,N}
  return convert(Array{T,N}, collect(a))
end

function Base.size(a::KroneckerArray)
  return ntuple(dim -> size(arg1(a), dim) * size(arg2(a), dim), ndims(a))
end

function Base.axes(a::KroneckerArray)
  return ntuple(ndims(a)) do dim
    return CartesianProductUnitRange(
      axes(arg1(a), dim) × axes(arg2(a), dim), Base.OneTo(size(a, dim))
    )
  end
end

arguments(a::KroneckerArray) = (arg1(a), arg2(a))
arguments(a::KroneckerArray, n::Int) = arguments(a)[n]
argument_types(a::KroneckerArray) = argument_types(typeof(a))
argument_types(::Type{<:KroneckerArray{<:Any,<:Any,A,B}}) where {A,B} = (A, B)

function Base.print_array(io::IO, a::KroneckerArray)
  Base.print_array(io, arg1(a))
  println(io, "\n ⊗")
  Base.print_array(io, arg2(a))
  return nothing
end
function Base.show(io::IO, a::KroneckerArray)
  show(io, arg1(a))
  print(io, " ⊗ ")
  show(io, arg2(a))
  return nothing
end

⊗(a::AbstractArray, b::AbstractArray) = KroneckerArray(a, b)
⊗(a::Number, b::Number) = a * b
⊗(a::Number, b::AbstractArray) = a * b
⊗(a::AbstractArray, b::Number) = a * b

function Base.getindex(a::KroneckerArray, i::Integer)
  return a[CartesianIndices(a)[i]]
end

using GPUArraysCore: GPUArraysCore
function Base.getindex(a::KroneckerArray{<:Any,N}, I::Vararg{Integer,N}) where {N}
  GPUArraysCore.assertscalar("getindex")
  I′ = ntuple(Val(N)) do dim
    return cartesianproduct(axes(a, dim))[I[dim]]
  end
  return a[I′...]
end

# Indexing logic.
function Base.to_indices(
  a::KroneckerArray, inds, I::Tuple{Union{CartesianPair,CartesianProduct},Vararg}
)
  I1 = to_indices(arg1(a), arg1.(inds), arg1.(I))
  I2 = to_indices(arg2(a), arg2.(inds), arg2.(I))
  return I1 .× I2
end

# Allow customizing for `FillArrays.Eye`.
_getindex(a::AbstractArray, I...) = a[I...]
function Base.getindex(
  a::KroneckerArray{<:Any,N}, I::Vararg{Union{CartesianPair,CartesianProduct},N}
) where {N}
  I′ = to_indices(a, I)
  return _getindex(arg1(a), arg1.(I′)...) ⊗ _getindex(arg2(a), arg2.(I′)...)
end
# Fix ambigiuity error.
Base.getindex(a::KroneckerArray{<:Any,0}) = arg1(a)[] * arg2(a)[]

# Allow customizing for `FillArrays.Eye`.
_view(a::AbstractArray, I...) = view(a, I...)
arg1(::Colon) = (:)
arg2(::Colon) = (:)
arg1(::Base.Slice) = (:)
arg2(::Base.Slice) = (:)
function Base.view(
  a::KroneckerArray{<:Any,N},
  I::Vararg{Union{CartesianProduct,CartesianProductUnitRange,Base.Slice,Colon},N},
) where {N}
  return _view(arg1(a), arg1.(I)...) ⊗ _view(arg2(a), arg2.(I)...)
end
function Base.view(a::KroneckerArray{<:Any,N}, I::Vararg{CartesianPair,N}) where {N}
  return _view(arg1(a), arg1.(I)...) ⊗ _view(arg2(a), arg2.(I)...)
end
# Fix ambigiuity error.
Base.view(a::KroneckerArray{<:Any,0}) = _view(arg1(a)) * _view(arg2(a))

function Base.:(==)(a::KroneckerArray, b::KroneckerArray)
  return arg1(a) == arg1(b) && arg2(a) == arg2(b)
end
function Base.isapprox(a::KroneckerArray, b::KroneckerArray; kwargs...)
  return isapprox(arg1(a), arg1(b); kwargs...) && isapprox(arg2(a), arg2(b); kwargs...)
end
function Base.iszero(a::KroneckerArray)
  return iszero(arg1(a)) || iszero(arg2(a))
end
function Base.isreal(a::KroneckerArray)
  return isreal(arg1(a)) && isreal(arg2(a))
end

using DiagonalArrays: DiagonalArrays, diagonal
function DiagonalArrays.diagonal(a::KroneckerArray)
  return diagonal(arg1(a)) ⊗ diagonal(arg2(a))
end

Base.real(a::KroneckerArray{<:Real}) = a
function Base.real(a::KroneckerArray)
  if iszero(imag(arg1(a))) || iszero(imag(arg2(a)))
    return real(arg1(a)) ⊗ real(arg2(a))
  elseif iszero(real(arg1(a))) || iszero(real(arg2(a)))
    return -imag(arg1(a)) ⊗ imag(arg2(a))
  end
  return real(arg1(a)) ⊗ real(arg2(a)) - imag(arg1(a)) ⊗ imag(arg2(a))
end
Base.imag(a::KroneckerArray{<:Real}) = zero(a)
function Base.imag(a::KroneckerArray)
  if iszero(imag(arg1(a))) || iszero(real(arg2(a)))
    return real(arg1(a)) ⊗ imag(arg2(a))
  elseif iszero(real(arg1(a))) || iszero(imag(arg2(a)))
    return imag(arg1(a)) ⊗ real(arg2(a))
  end
  return real(arg1(a)) ⊗ imag(arg2(a)) + imag(arg1(a)) ⊗ real(arg2(a))
end

for f in [:transpose, :adjoint, :inv]
  @eval begin
    function Base.$f(a::KroneckerArray)
      return $f(arg1(a)) ⊗ $f(arg2(a))
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
  bc_a = Broadcasted(A, bc.f, arg1.(bc.args), arg1.(ax))
  bc_b = Broadcasted(B, bc.f, arg2.(bc.args), arg2.(ax))
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

# Compatibility with MapBroadcast.jl.
using MapBroadcast: MapBroadcast, MapFunction
function Base.broadcasted(
  style::KroneckerStyle, f::MapFunction{typeof(*),<:Tuple{<:Number,MapBroadcast.Arg}}, a
)
  return broadcasted(style, *, f.args[1], a)
end
function Base.broadcasted(
  style::KroneckerStyle, f::MapFunction{typeof(*),<:Tuple{MapBroadcast.Arg,<:Number}}, a
)
  return broadcasted(style, *, a, f.args[2])
end
function Base.broadcasted(
  style::KroneckerStyle, f::MapFunction{typeof(/),<:Tuple{MapBroadcast.Arg,<:Number}}, a
)
  return broadcasted(style, /, a, f.args[2])
end
