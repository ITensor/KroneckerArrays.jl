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

function Base.:*(a::Number, b::EyeKronecker)
  return b.a ⊗ (a * b.b)
end
function Base.:*(a::Number, b::KroneckerEye)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::Number, b::EyeEye)
  return error("Can't multiply `Eye ⊗ Eye` by a number.")
end
function Base.:*(a::EyeKronecker, b::Number)
  return a.a ⊗ (a.b * b)
end
function Base.:*(a::KroneckerEye, b::Number)
  return (a.a * b) ⊗ a.b
end
function Base.:*(a::EyeEye, b::Number)
  return error("Can't multiply `Eye ⊗ Eye` by a number.")
end

function Base.:-(a::EyeKronecker)
  return a.a ⊗ (-a.b)
end
function Base.:-(a::KroneckerEye)
  return (-a.a) ⊗ a.b
end
function Base.:-(a::EyeEye)
  return error("Can't multiply `Eye ⊗ Eye` by a number.")
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

function Base.map!(f::typeof(identity), dest::EyeKronecker, src::EyeKronecker)
  map!(f, dest.b, src.b)
  return dest
end
function Base.map!(f::typeof(identity), dest::KroneckerEye, src::KroneckerEye)
  map!(f, dest.a, src.a)
  return dest
end
function Base.map!(::typeof(identity), dest::EyeEye, src::EyeEye)
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
  map!(f, dest.b, a.b)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  map!(f, dest.a, a.a)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeKronecker, a::EyeKronecker)
  map!(f, dest.b, a.b)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  map!(f, dest.a, a.a)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
