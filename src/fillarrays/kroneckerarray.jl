using FillArrays: FillArrays, Ones, Zeros
function FillArrays.fillsimilar(
  a::Zeros{T},
  ax::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
) where {T}
  return Zeros{T}(arg1.(ax)) ⊗ Zeros{T}(arg2.(ax))
end

using FillArrays: RectDiagonal, OnesVector
const RectEye{T,V<:OnesVector{T},Axes} = RectDiagonal{T,V,Axes}

using FillArrays: Eye
const EyeKronecker{T,A<:Eye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerEye{T,A<:AbstractMatrix{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}
const EyeEye{T,A<:Eye{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}

using FillArrays: SquareEye
const SquareEyeKronecker{T,A<:SquareEye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerSquareEye{T,A<:AbstractMatrix{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}
const SquareEyeSquareEye{T,A<:SquareEye{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}

using DiagonalArrays: Delta
const DeltaKronecker{T,N,A<:Delta{T,N},B<:AbstractArray{T,N}} = KroneckerArray{T,N,A,B}
const KroneckerDelta{T,N,A<:AbstractArray{T,N},B<:Delta{T,N}} = KroneckerArray{T,N,A,B}
const DeltaDelta{T,N,A<:Delta{T,N},B<:Delta{T,N}} = KroneckerArray{T,N,A,B}

_getindex(a::Eye, I1::Colon, I2::Colon) = a
_getindex(a::Eye, I1::Base.Slice, I2::Base.Slice) = a
_getindex(a::Eye, I1::Base.Slice, I2::Colon) = a
_getindex(a::Eye, I1::Colon, I2::Base.Slice) = a
_view(a::Eye, I1::Colon, I2::Colon) = a
_view(a::Eye, I1::Base.Slice, I2::Base.Slice) = a
_view(a::Eye, I1::Base.Slice, I2::Colon) = a
_view(a::Eye, I1::Colon, I2::Base.Slice) = a

function _getindex(a::Delta, I1::Union{Colon,Base.Slice}, Irest::Union{Colon,Base.Slice}...)
  return a
end
function _view(a::Delta, I1::Union{Colon,Base.Slice}, Irest::Union{Colon,Base.Slice}...)
  return a
end

# Like `adapt` but preserves `Eye`.
_adapt(to, a::Eye) = a
_adapt(to, a::Delta) = a

# Allows customizing for `FillArrays.Eye`.
function _convert(::Type{AbstractArray{T}}, a::RectDiagonal) where {T}
  return _convert(AbstractMatrix{T}, a)
end
function _convert(::Type{AbstractMatrix{T}}, a::RectDiagonal) where {T}
  return RectDiagonal(convert(AbstractVector{T}, _diagview(a)), axes(a))
end

# Like `similar` but preserves `Eye`.
function _similar(a::AbstractArray, elt::Type, ax::Tuple)
  return similar(a, elt, ax)
end
function _similar(A::Type{<:AbstractArray}, ax::Tuple)
  return similar(A, ax)
end
function _similar(a::AbstractArray, ax::Tuple)
  return _similar(a, eltype(a), ax)
end
function _similar(a::AbstractArray, elt::Type)
  return _similar(a, elt, axes(a))
end
function _similar(a::AbstractArray)
  return _similar(a, eltype(a), axes(a))
end

# Like `similar` but preserves `Eye`.
function _similar(a::Eye, elt::Type, axs::NTuple{2,AbstractUnitRange})
  return Eye{elt}(axs)
end
function _similar(arrayt::Type{<:Eye}, axs::NTuple{2,AbstractUnitRange})
  return Eye{eltype(arrayt)}(axs)
end

# Like `similar` but preserves `SquareEye`.
function _similar(a::SquareEye, elt::Type, axs::NTuple{2,AbstractUnitRange})
  return Eye{elt}((only(unique(axs)),))
end
function _similar(arrayt::Type{<:SquareEye}, axs::NTuple{2,AbstractUnitRange})
  return Eye{eltype(arrayt)}((only(unique(axs)),))
end

function _similar(a::Delta, elt::Type, axs::Tuple{Vararg{AbstractUnitRange}})
  return Delta{elt}(axs)
end
function _similar(arrayt::Type{<:Delta}, axs::Tuple{Vararg{AbstractUnitRange}})
  return Delta{eltype(arrayt)}(axs)
end

# Like `copy` but preserves `Eye`/`Delta`.
_copy(a::Eye) = a
_copy(a::Delta) = a

function _copyto!!(dest::Eye{<:Any,N}, src::Eye{<:Any,N}) where {N}
  size(dest) == size(src) ||
    throw(ArgumentError("Sizes do not match: $(size(dest)) != $(size(src))."))
  return dest
end
function _copyto!!(dest::Delta{<:Any,N}, src::Delta{<:Any,N}) where {N}
  size(dest) == size(src) ||
    throw(ArgumentError("Sizes do not match: $(size(dest)) != $(size(src))."))
  return dest
end

# TODO: Define `DerivableInterfaces.permuteddims` and overload that instead.
function Base.PermutedDimsArray(a::Delta, perm)
  ax_perm = Base.PermutedDimsArrays.genperm(axes(a), perm)
  return Delta{eltype(a)}(ax_perm)
end

function _permutedims!!(dest::Delta, src::Delta, perm)
  Base.PermutedDimsArrays.genperm(axes(src), perm) == axes(dest) ||
    throw(ArgumentError("Permuted axes do not match."))
  return dest
end

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

function DerivableInterfaces.zero!(a::DeltaKronecker)
  zero!(a.b)
  return a
end
function DerivableInterfaces.zero!(a::KroneckerDelta)
  zero!(a.a)
  return a
end
function DerivableInterfaces.zero!(a::DeltaDelta)
  return throw(ArgumentError("Can't zero out `Delta ⊗ Delta`."))
end

using Base.Broadcast:
  AbstractArrayStyle, AbstractArrayStyle, BroadcastStyle, Broadcasted, broadcasted

struct EyeStyle <: AbstractArrayStyle{2} end
EyeStyle(::Val{2}) = EyeStyle()
function _BroadcastStyle(::Type{<:Eye})
  return EyeStyle()
end
Base.BroadcastStyle(style1::EyeStyle, style2::EyeStyle) = EyeStyle()
Base.BroadcastStyle(style1::EyeStyle, style2::DefaultArrayStyle) = style2

function _copyto!!(dest::Eye, src::Broadcasted{<:EyeStyle,<:Any,typeof(identity)})
  axes(dest) == axes(src) || error("Dimension mismatch.")
  return dest
end

function Base.similar(bc::Broadcasted{EyeStyle}, elt::Type)
  return Eye{elt}(axes(bc))
end

# TODO: Define in terms of `_copyto!!` that is called on each argument.
function Base.copyto!(dest::EyeKronecker, a::Sum{<:KroneckerStyle{<:Any,EyeStyle()}})
  dest2 = arg2(dest)
  f = LinearCombination(a)
  args = arguments(a)
  arg2s = arg2.(args)
  dest2 .= f.(arg2s...)
  return dest
end
function Base.copyto!(dest::KroneckerEye, a::Sum{<:KroneckerStyle{<:Any,<:Any,EyeStyle()}})
  dest1 = arg1(dest)
  f = LinearCombination(a)
  args = arguments(a)
  arg1s = arg1.(args)
  dest1 .= f.(arg1s...)
  return dest
end
function Base.copyto!(dest::EyeEye, a::Sum{<:KroneckerStyle{<:Any,EyeStyle(),EyeStyle()}})
  return error("Can't write in-place to `Eye ⊗ Eye`.")
end

struct DeltaStyle{N} <: AbstractArrayStyle{N} end
DeltaStyle(::Val{N}) where {N} = DeltaStyle{N}()
DeltaStyle{M}(::Val{N}) where {M,N} = DeltaStyle{N}()
function _BroadcastStyle(A::Type{<:Delta})
  return DeltaStyle{ndims(A)}()
end
Base.BroadcastStyle(style1::DeltaStyle, style2::DeltaStyle) = DeltaStyle()
Base.BroadcastStyle(style1::DeltaStyle, style2::DefaultArrayStyle) = style2

function _copyto!!(dest::Delta, src::Broadcasted{<:DeltaStyle,<:Any,typeof(identity)})
  axes(dest) == axes(src) || error("Dimension mismatch.")
  return dest
end

function Base.similar(bc::Broadcasted{<:DeltaStyle}, elt::Type)
  return Delta{elt}(axes(bc))
end

# TODO: Dispatch on `DeltaStyle`.
function Base.copyto!(dest::DeltaKronecker, a::Sum{<:KroneckerStyle})
  dest2 = arg2(dest)
  f = LinearCombination(a)
  args = arguments(a)
  arg2s = arg2.(args)
  dest2 .= f.(arg2s...)
  return dest
end
# TODO: Dispatch on `DeltaStyle`.
function Base.copyto!(dest::KroneckerDelta, a::Sum{<:KroneckerStyle})
  dest1 = arg1(dest)
  f = LinearCombination(a)
  args = arguments(a)
  arg1s = arg1.(args)
  dest1 .= f.(arg1s...)
  return dest
end
# TODO: Dispatch on `DeltaStyle`.
function Base.copyto!(dest::DeltaDelta, a::Sum{<:KroneckerStyle})
  return error("Can't write in-place to `Delta ⊗ Delta`.")
end

# Simplification rules similar to those for FillArrays.jl:
# https://github.com/JuliaArrays/FillArrays.jl/blob/v1.13.0/src/fillbroadcast.jl
using FillArrays: Zeros
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(+),
  a::KroneckerArray,
  b::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
)
  # TODO: Promote the element types.
  return a
end
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(+),
  a::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
  b::KroneckerArray,
)
  # TODO: Promote the element types.
  return b
end
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(+),
  a::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
  b::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
)
  # TODO: Promote the element types and axes.
  return b
end
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(-),
  a::KroneckerArray,
  b::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
)
  # TODO: Promote the element types.
  return a
end
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(-),
  a::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
  b::KroneckerArray,
)
  # TODO: Promote the element types.
  # TODO: Return `broadcasted(-, b)`.
  return -b
end
function Base.broadcasted(
  style::KroneckerStyle,
  ::typeof(-),
  a::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
  b::KroneckerArray{<:Any,<:Any,<:Zeros,<:Zeros},
)
  # TODO: Promote the element types and axes.
  return b
end
