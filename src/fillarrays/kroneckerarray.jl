using FillArrays: FillArrays, Zeros
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

# Like `adapt` but preserves `Eye`.
_adapt(to, a::Eye) = a

# Allows customizing for `FillArrays.Eye`.
function _convert(::Type{AbstractArray{T}}, a::RectDiagonal) where {T}
  _convert(AbstractMatrix{T}, a)
end
function _convert(::Type{AbstractMatrix{T}}, a::RectDiagonal) where {T}
  RectDiagonal(convert(AbstractVector{T}, _diagview(a)), axes(a))
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

# Like `copy` but preserves `Eye`.
_copy(a::Eye) = a

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

using Base.Broadcast:
  AbstractArrayStyle, AbstractArrayStyle, BroadcastStyle, Broadcasted, broadcasted

struct EyeStyle <: AbstractArrayStyle{2} end
EyeStyle(::Val{2}) = EyeStyle()
function _BroadcastStyle(::Type{<:Eye})
  return EyeStyle()
end
Base.BroadcastStyle(style1::EyeStyle, style2::EyeStyle) = EyeStyle()
Base.BroadcastStyle(style1::EyeStyle, style2::DefaultArrayStyle) = style2

function Base.similar(bc::Broadcasted{EyeStyle}, elt::Type)
  return Eye{elt}(axes(bc))
end

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
