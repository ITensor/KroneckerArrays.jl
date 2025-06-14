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
