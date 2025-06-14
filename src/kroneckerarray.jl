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

for f in [:transpose, :adjoint, :inv]
  @eval begin
    function Base.$f(a::KroneckerArray)
      return $f(a.a) ⊗ $f(a.b)
    end
  end
end

function Base.:*(a::Number, b::KroneckerArray)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::KroneckerArray, b::Number)
  return a.a ⊗ (a.b * b)
end
function Base.:/(a::KroneckerArray, b::Number)
  return a.a ⊗ (a.b / b)
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
      else
        throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
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

function _map!!(f::F, dest::AbstractArray, srcs::AbstractArray...) where {F}
  map!(f, dest, srcs...)
  return dest
end

for f in [:identity, :conj]
  @eval begin
    function Base.map!(::typeof($f), dest::KroneckerArray, src::KroneckerArray)
      _map!!($f, dest.a, src.a)
      _map!!($f, dest.b, src.b)
      return dest
    end
  end
end

for f in [:+, :-]
  @eval begin
    function Base.map!(
      ::typeof($f), dest::KroneckerArray, a::KroneckerArray, b::KroneckerArray
    )
      if a.b == b.b
        map!($f, dest.a, a.a, b.a)
        map!(identity, dest.b, a.b)
        return dest
      elseif a.a == b.a
        map!(identity, dest.a, a.a)
        map!($f, dest.b, a.b, b.b)
        return dest
      else
        throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or second arguments match.",
          ),
        )
      end
    end
  end
end

function Base.map!(
  f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerArray, src::KroneckerArray
)
  map!(f, dest.a, src.a)
  map!(identity, dest.b, src.b)
  return dest
end

for op in [:*, :/]
  @eval begin
    function Base.map!(
      f::Base.Fix2{typeof($op),<:Number}, dest::KroneckerArray, src::KroneckerArray
    )
      map!(identity, dest.a, src.a)
      map!(f, dest.b, src.b)
      return dest
    end
  end
end

using DiagonalArrays: DiagonalArrays, diagonal
function DiagonalArrays.diagonal(a::KroneckerArray)
  return diagonal(a.a) ⊗ diagonal(a.b)
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
