using FillArrays: Eye, SquareEye
using LinearAlgebra: LinearAlgebra, mul!, pinv

function check_mul_axes(a::AbstractMatrix, b::AbstractMatrix)
  return axes(a, 2) == axes(b, 1) || throw(DimensionMismatch("Incompatible matrix sizes."))
end

function _mul(a::Eye, b::Eye)
  check_mul_axes(a, b)
  T = promote_type(eltype(a), eltype(b))
  return Eye{T}((axes(a, 1), axes(b, 2)))
end
function _mul(a::SquareEye, b::SquareEye)
  check_mul_axes(a, b)
  return Diagonal(diagview(a) .* diagview(b))
end

for f in MATRIX_FUNCTIONS
  @eval begin
    function Base.$f(a::EyeKronecker)
      LinearAlgebra.checksquare(a.a)
      return a.a ⊗ $f(a.b)
    end
    function Base.$f(a::KroneckerEye)
      LinearAlgebra.checksquare(a.b)
      return $f(a.a) ⊗ a.b
    end
    function Base.$f(a::EyeEye)
      LinearAlgebra.checksquare(a)
      return throw(ArgumentError("`$($f)` on `Eye ⊗ Eye` is not supported."))
    end
  end
end

function LinearAlgebra.mul!(
  c::EyeKronecker, a::EyeKronecker, b::EyeKronecker, α::Number, β::Number
)
  iszero(β) ||
    iszero(c) ||
    throw(
      ArgumentError(
        "Can't multiple KroneckerArrays with nonzero β and nonzero destination."
      ),
    )
  check_mul_axes(a.a, b.a)
  mul!(c.b, a.b, b.b, α, β)
  return c
end
function LinearAlgebra.mul!(
  c::KroneckerEye, a::KroneckerEye, b::KroneckerEye, α::Number, β::Number
)
  iszero(β) ||
    iszero(c) ||
    throw(
      ArgumentError(
        "Can't multiple KroneckerArrays with nonzero β and nonzero destination."
      ),
    )
  check_mul_axes(a.b, b.b)
  mul!(c.a, a.a, b.a, α, β)
  return c
end
function LinearAlgebra.mul!(c::EyeEye, a::EyeEye, b::EyeEye, α::Number, β::Number)
  return throw(ArgumentError("Can't multiple `Eye ⊗ Eye` in-place."))
end

function LinearAlgebra.pinv(a::EyeKronecker; kwargs...)
  return a.a ⊗ pinv(a.b; kwargs...)
end
function LinearAlgebra.pinv(a::KroneckerEye; kwargs...)
  return pinv(a.a; kwargs...) ⊗ a.b
end
function LinearAlgebra.pinv(a::EyeEye; kwargs...)
  return a
end
