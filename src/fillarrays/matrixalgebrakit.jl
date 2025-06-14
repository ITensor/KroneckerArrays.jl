####################################################################################
# Special cases for MatrixAlgebraKit factorizations of `Eye(n) ⊗ A` and
# `A ⊗ Eye(n)` where `A`.
# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/34
# is merged.

struct SquareEyeAlgorithm{KWargs<:NamedTuple} <: AbstractAlgorithm
  kwargs::KWargs
end
SquareEyeAlgorithm(; kwargs...) = SquareEyeAlgorithm((; kwargs...))

# Defined to avoid type piracy.
_copy_input_squareeye(f::F, a) where {F} = copy_input(f, a)
_copy_input_squareeye(f::F, a::SquareEye) where {F} = a

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
]
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.copy_input(::typeof($f), a::$T)
        return _copy_input_squareeye($f, a.a) ⊗ _copy_input_squareeye($f, a.b)
      end
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
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a; kwargs...) = $f(a; kwargs...)
    $f′(a::Type{<:SquareEye}; kwargs...) = SquareEyeAlgorithm(; kwargs...)
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.$f(A::Type{<:$T}; kwargs1=(;), kwargs2=(;), kwargs...)
        A1, A2 = argument_types(A)
        return KroneckerAlgorithm(
          $f′(A1; kwargs..., kwargs1...), $f′(A2; kwargs..., kwargs2...)
        )
      end
    end
  end
end

# Defined to avoid type piracy.
_initialize_output_squareeye(f::F, a) where {F} = initialize_output(f, a)
_initialize_output_squareeye(f::F, a, alg) where {F} = initialize_output(f, a, alg)

for f in [:left_null!, :right_null!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = a
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = a
  end
end
for f in [
  :qr_compact!,
  :qr_full!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_orth!,
  :right_polar!,
]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = (a, a)
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, a)
  end
end
_initialize_output_squareeye(::typeof(eig_full!), a::SquareEye) = complex.((a, a))
_initialize_output_squareeye(::typeof(eig_full!), a::SquareEye, alg) = complex.((a, a))
_initialize_output_squareeye(::typeof(eigh_full!), a::SquareEye) = (real(a), a)
_initialize_output_squareeye(::typeof(eigh_full!), a::SquareEye, alg) = (real(a), a)
for f in [:svd_compact!, :svd_full!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = (a, real(a), a)
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, real(a), a)
  end
end

for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_orth!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_orth!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F, alg; kwargs...) = $f(a, F, alg; kwargs...)
    $f′(a, F, alg::SquareEyeAlgorithm) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(::typeof($f), a::$T)
        return _initialize_output_squareeye($f, a.a) .⊗
               _initialize_output_squareeye($f, a.b)
      end
      function MatrixAlgebraKit.initialize_output(
        ::typeof($f), a::$T, alg::KroneckerAlgorithm
      )
        return _initialize_output_squareeye($f, a.a, alg.a) .⊗
               _initialize_output_squareeye($f, a.b, alg.b)
      end
      function MatrixAlgebraKit.$f(
        a::$T, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
      )
        $f′(a.a, Base.Fix2(getfield, :a).(F), alg.a; kwargs..., kwargs1...)
        $f′(a.b, Base.Fix2(getfield, :b).(F), alg.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

for f in [:left_null!, :right_null!]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F; kwargs...) = $f(a, F; kwargs...)
    $f′(a::SquareEye, F) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(::typeof($f), a::$T)
        return _initialize_output_squareeye($f, a.a) ⊗ _initialize_output_squareeye($f, a.b)
      end
      function MatrixAlgebraKit.$f(a::$T, F; kwargs1=(;), kwargs2=(;), kwargs...)
        $f′(a.a, F.a; kwargs..., kwargs1...)
        $f′(a.b, F.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

function MatrixAlgebraKit.initialize_output(f::typeof(left_null!), a::SquareEyeSquareEye)
  return _initialize_output_squareeye(f, a.a) ⊗ _initialize_output_squareeye(f, a.b)
end
function MatrixAlgebraKit.left_null!(
  a::SquareEyeSquareEye, F; kwargs1=(;), kwargs2=(;), kwargs...
)
  return throw(MethodError(left_null!, (a, F)))
end

function MatrixAlgebraKit.initialize_output(f::typeof(right_null!), a::SquareEyeSquareEye)
  return _initialize_output_squareeye(f, a.a) ⊗ _initialize_output_squareeye(f, a.b)
end
function MatrixAlgebraKit.right_null!(
  a::SquareEyeSquareEye, F; kwargs1=(;), kwargs2=(;), kwargs...
)
  return throw(MethodError(right_null!, (a, F)))
end

_initialize_output_squareeye(::typeof(eig_vals!), a::SquareEye) = parent(a)
_initialize_output_squareeye(::typeof(eig_vals!), a::SquareEye, alg) = parent(a)
for f in [:eigh_vals!, svd_vals!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye) = real(parent(a))
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = real(parent(a))
  end
end

for f in [:eig_vals!, :eigh_vals!, :svd_vals!]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F, alg; kwargs...) = $f(a, F, alg; kwargs...)
    $f′(a, F, alg::SquareEyeAlgorithm) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(
        ::typeof($f), a::$T, alg::KroneckerAlgorithm
      )
        return _initialize_output_squareeye($f, a.a, alg.a) ⊗
               _initialize_output_squareeye($f, a.b, alg.b)
      end
      function MatrixAlgebraKit.$f(
        a::$T, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
      )
        $f′(a.a, F.a, alg.a; kwargs..., kwargs1...)
        $f′(a.b, F.b, alg.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

using MatrixAlgebraKit: TruncationStrategy, diagview, findtruncated, truncate!

struct KroneckerTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

# Avoid instantiating the identity.
function Base.getindex(a::SquareEyeKronecker, I::Vararg{CartesianProduct{Colon},2})
  return a.a ⊗ a.b[I[1].b, I[2].b]
end
function Base.getindex(a::KroneckerSquareEye, I::Vararg{CartesianProduct{<:Any,Colon},2})
  return a.a[I[1].a, I[2].a] ⊗ a.b
end
function Base.getindex(a::SquareEyeSquareEye, I::Vararg{CartesianProduct{Colon,Colon},2})
  return a
end

using FillArrays: OnesVector
const OnesKroneckerVector{T,A<:OnesVector{T},B<:AbstractVector{T}} = KroneckerVector{T,A,B}
const KroneckerOnesVector{T,A<:AbstractVector{T},B<:OnesVector{T}} = KroneckerVector{T,A,B}
const OnesVectorOnesVector{T,A<:OnesVector{T},B<:OnesVector{T}} = KroneckerVector{T,A,B}

function MatrixAlgebraKit.findtruncated(
  values::OnesKroneckerVector, strategy::KroneckerTruncationStrategy
)
  I = findtruncated(Vector(values), strategy.strategy)
  prods = collect(only(axes(values)).product)[I]
  I_data = unique(map(x -> x.a, prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> x.a == i, prods) == length(values.a)
  end
  return (:) × I_data
end
function MatrixAlgebraKit.findtruncated(
  values::KroneckerOnesVector, strategy::KroneckerTruncationStrategy
)
  I = findtruncated(Vector(values), strategy.strategy)
  prods = collect(only(axes(values)).product)[I]
  I_data = unique(map(x -> x.b, prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> x.b == i, prods) == length(values.b)
  end
  return I_data × (:)
end
function MatrixAlgebraKit.findtruncated(
  values::OnesVectorOnesVector, strategy::KroneckerTruncationStrategy
)
  return throw(ArgumentError("Can't truncate Eye ⊗ Eye."))
end

for f in [:eig_trunc!, :eigh_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f), DV::NTuple{2,KroneckerMatrix}, strategy::TruncationStrategy
    )
      return truncate!($f, DV, KroneckerTruncationStrategy(strategy))
    end
    function MatrixAlgebraKit.truncate!(
      ::typeof($f), (D, V)::NTuple{2,KroneckerMatrix}, strategy::KroneckerTruncationStrategy
    )
      I = findtruncated(diagview(D), strategy)
      return (D[I, I], V[(:) × (:), I])
    end
  end
end

function MatrixAlgebraKit.truncate!(
  f::typeof(svd_trunc!), USVᴴ::NTuple{3,KroneckerMatrix}, strategy::TruncationStrategy
)
  return truncate!(f, USVᴴ, KroneckerTruncationStrategy(strategy))
end
function MatrixAlgebraKit.truncate!(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::NTuple{3,KroneckerMatrix},
  strategy::KroneckerTruncationStrategy,
)
  I = findtruncated(diagview(S), strategy)
  return (U[(:) × (:), I], S[I, I], Vᴴ[I, (:) × (:)])
end
