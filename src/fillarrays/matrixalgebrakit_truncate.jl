using MatrixAlgebraKit: TruncationStrategy, diagview, findtruncated, truncate!

struct KroneckerTruncationStrategy{T<:TruncationStrategy} <: TruncationStrategy
  strategy::T
end

# Avoid instantiating the identity.
function Base.getindex(a::EyeKronecker, I::Vararg{CartesianProduct{Colon},2})
  return a.a ⊗ a.b[I[1].b, I[2].b]
end
function Base.getindex(a::KroneckerEye, I::Vararg{CartesianProduct{<:Any,Colon},2})
  return a.a[I[1].a, I[2].a] ⊗ a.b
end
function Base.getindex(a::EyeEye, I::Vararg{CartesianProduct{Colon,Colon},2})
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
  prods = only(axes(values))[I]
  I_data = unique(arg1.(prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> arg1(x) == i, prods) == length(arg1(values))
  end
  return (:) × I_data
end
function MatrixAlgebraKit.findtruncated(
  values::KroneckerOnesVector, strategy::KroneckerTruncationStrategy
)
  I = findtruncated(Vector(values), strategy.strategy)
  prods = only(axes(values))[I]
  I_data = unique(map(x -> arg2(x), prods))
  # Drop truncations that occur within the identity.
  I_data = filter(I_data) do i
    return count(x -> arg2(x) == i, prods) == length(arg2(values))
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
