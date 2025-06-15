function infimum(r1::AbstractRange, r2::AbstractUnitRange)
  Base.require_one_based_indexing(r1, r2)
  if length(r1) ≤ length(r2)
    return r1
  else
    return r2
  end
end
function supremum(r1::AbstractRange, r2::AbstractUnitRange)
  Base.require_one_based_indexing(r1, r2)
  if length(r1) ≥ length(r2)
    return r1
  else
    return r2
  end
end

# Allow customization for `Eye`.
_diagview(a::Eye) = parent(a)

function _copy_input(f::F, a::Eye) where {F}
  return a
end

struct EyeAlgorithm{KWargs<:NamedTuple} <: AbstractAlgorithm
  kwargs::KWargs
end
EyeAlgorithm(; kwargs...) = EyeAlgorithm((; kwargs...))

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  _f = Symbol(:_, f)
  @eval begin
    function $_f(A::Type{<:Eye}; kwargs...)
      return EyeAlgorithm(; kwargs...)
    end
  end
end

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
  :svd_vals,
]
  f! = Symbol(f, "!")
  @eval begin
    function MatrixAlgebraKit.$f!(a::Eye, F, ::EyeAlgorithm)
      return F
    end
  end
end

_complex(a::AbstractArray) = complex(a)
_complex(a::Eye{<:Complex}) = a
_complex(a::Eye) = _similar(a, complex(eltype(a)))
_real(a::AbstractArray) = real(a)
_real(a::Eye{<:Real}) = a
_real(a::Eye) = _similar(a, real(eltype(a)))

# Implementations of `Eye` factorizations are doing in `initialize_output`
# so they can be used in KroneckerArray factorizations.
function _initialize_output(::typeof(eig_full!), a::Eye, ::EyeAlgorithm)
  LinearAlgebra.checksquare(a)
  return _complex.((a, a))
end
function _initialize_output(::typeof(eigh_full!), a::Eye, ::EyeAlgorithm)
  LinearAlgebra.checksquare(a)
  return (_real(a), a)
end
function _initialize_output(::typeof(eig_vals!), a::Eye, ::EyeAlgorithm)
  LinearAlgebra.checksquare(a)
  # TODO: Use `_diagview`/`_diag`.
  return _complex(parent(a))
end
function _initialize_output(::typeof(eigh_vals!), a::Eye, ::EyeAlgorithm)
  LinearAlgebra.checksquare(a)
  # TODO: Use `_diagview`/`_diag`.
  return _real(parent(a))
end
function _initialize_output(::typeof(svd_compact!), a::Eye, ::EyeAlgorithm)
  ax_s = (infimum(axes(a)...), infimum(reverse(axes(a))...))
  ax_u = (axes(a, 1), ax_s[2])
  ax_v = (ax_s[1], axes(a, 2))
  Tr = real(eltype(a))
  return (_similar(a, ax_u), _similar(a, Tr, ax_s), _similar(a, ax_v))
end
function _initialize_output(::typeof(svd_full!), a::Eye, ::EyeAlgorithm)
  ax_s = axes(a)
  ax_u = (axes(a, 1), axes(a, 1))
  ax_v = (axes(a, 2), axes(a, 2))
  Tr = real(eltype(a))
  return (_similar(a, ax_u), _similar(a, Tr, ax_s), _similar(a, ax_v))
end
function _initialize_output(::typeof(svd_vals!), a::Eye, ::EyeAlgorithm)
  # TODO: Use `_diagview`/`_diag`.
  return _real(parent(a))
end

for f in [:left_polar!, :right_polar!, qr_compact!, lq_compact!]
  @eval begin
    function _initialize_output(::typeof($f), a::Eye, ::EyeAlgorithm)
      ax = infimum(axes(a)...)
      ax_x = (axes(a, 1), ax)
      ax_y = (ax, axes(a, 2))
      return (_similar(a, ax_x), _similar(a, ax_y))
    end
  end
end

for f in [qr_full!, lq_full!]
  @eval begin
    function _initialize_output(::typeof($f), a::Eye, ::EyeAlgorithm)
      ax = supremum(axes(a)...)
      ax_x = (axes(a, 1), ax)
      ax_y = (ax, axes(a, 2))
      return (_similar(a, ax_x), _similar(a, ax_y))
    end
  end
end

for f in [:left_orth!, :right_orth!]
  @eval begin
    function _initialize_output(::typeof($f), a::Eye)
      ax = infimum(axes(a)...)
      ax_x = (axes(a, 1), ax)
      ax_y = (ax, axes(a, 2))
      return (_similar(a, ax_x), _similar(a, ax_y))
    end
  end
end

for f in [:left_null!, :right_null!]
  _f = Symbol(:_, f)
  @eval begin
    function _initialize_output(::typeof($f), a::Eye)
      return a
    end
    function $_f(a::Eye, F)
      return F
    end

    function MatrixAlgebraKit.$f(a::EyeEye, F; kwargs...)
      return throw(MethodError($f, (a, F)))
    end
  end
end
