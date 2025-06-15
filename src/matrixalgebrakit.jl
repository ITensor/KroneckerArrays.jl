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

using MatrixAlgebraKit: MatrixAlgebraKit, diagview
function MatrixAlgebraKit.diagview(a::KroneckerMatrix)
  return diagview(a.a) ⊗ diagview(a.b)
end

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

function _copy_input(f::F, a::AbstractMatrix) where {F}
  return copy_input(f, a)
end

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
      return _copy_input($f, a.a) ⊗ _copy_input($f, a.b)
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
  _f = Symbol(:_, f)
  @eval begin
    function $_f(A::Type{<:AbstractMatrix}; kwargs...)
      return $f(A; kwargs...)
    end
    function MatrixAlgebraKit.$f(
      A::Type{<:KroneckerMatrix}; kwargs1=(;), kwargs2=(;), kwargs...
    )
      A1, A2 = argument_types(A)
      return KroneckerAlgorithm(
        $_f(A1; kwargs..., kwargs1...), $_f(A2; kwargs..., kwargs2...)
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

# Allows overloading while avoiding type piracy.
function _initialize_output(f::F, a::AbstractMatrix, alg::AbstractAlgorithm) where {F}
  return initialize_output(f, a, alg)
end
_initialize_output(f::F, a::AbstractMatrix) where {F} = initialize_output(f, a)

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
      return _initialize_output($f, a.a, alg.a) .⊗ _initialize_output($f, a.b, alg.b)
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
      return _initialize_output($f, a.a, alg.a) ⊗ _initialize_output($f, a.b, alg.b)
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
      return _initialize_output($f, a.a) .⊗ _initialize_output($f, a.b)
    end
  end
end

for f in [:left_null!, :right_null!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
      return _initialize_output($f, a.a) ⊗ _initialize_output($f, a.b)
    end
    function MatrixAlgebraKit.$f(a::KroneckerMatrix, F; kwargs1=(;), kwargs2=(;), kwargs...)
      $f(a.a, F.a; kwargs..., kwargs1...)
      $f(a.b, F.b; kwargs..., kwargs2...)
      return F
    end
  end
end
