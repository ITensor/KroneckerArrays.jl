using MatrixAlgebraKit:
    MatrixAlgebraKit,
    AbstractAlgorithm, TruncationStrategy,
    eig_full!, eig_full, eig_trunc!, eig_trunc, eig_vals!, eig_vals,
    eigh_full!, eigh_full, eigh_trunc!, eigh_trunc, eigh_vals!, eigh_vals,
    initialize_output,
    left_null!, left_null, left_orth!, left_orth, left_polar!, left_polar,
    lq_compact!, lq_compact, lq_full!, lq_full,
    qr_compact!, qr_compact, qr_full!, qr_full,
    right_null!, right_null, right_orth!, right_orth, right_polar!, right_polar,
    svd_compact!, svd_compact, svd_full!, svd_full, svd_trunc!, svd_trunc, svd_vals!, svd_vals,
    truncate
using MatrixAlgebraKit:
    eig_full, eig_vals, eigh_full, eigh_vals,
    qr_compact, qr_full,
    left_null, left_orth, left_polar,
    lq_compact, lq_full,
    right_null, right_orth, right_polar,
    svd_compact, svd_full
using MatrixAlgebraKit: TruncationStrategy, findtruncated, truncate
import MatrixAlgebraKit as MAK

DiagonalArrays.diagview(a::AbstractKroneckerMatrix) = ⊗(DiagonalArrays.diagview.(kroneckerfactors(a))...)
MatrixAlgebraKit.diagview(a::AbstractKroneckerMatrix) = DiagonalArrays.diagview(a)

struct KroneckerAlgorithm{A, B} <: AbstractAlgorithm
    a::A
    b::B
end

kroneckerfactors(alg::KroneckerAlgorithm) = alg.a, alg.b
kroneckerfactortypes(::Type{KroneckerAlgorithm{A, B}}) where {A, B} = (A, B)

for f in (
        :eig_full, :eigh_full,
        :qr_compact, :qr_full,
        :lq_compact, :lq_full,
        :left_polar, :right_polar,
        :svd_compact, :svd_full,
    )
    @eval MAK.copy_input(::typeof($f), a::AbstractKroneckerMatrix) =
        ⊗(MAK.copy_input.(($f,), kroneckerfactors(a))...)
end

for f in (
        :default_eig_algorithm, :default_eigh_algorithm,
        :default_lq_algorithm, :default_qr_algorithm,
        :default_polar_algorithm, :default_svd_algorithm,
    )
    @eval function MAK.$f(A::Type{<:AbstractKroneckerMatrix}; kwargs1 = (;), kwargs2 = (;), kwargs...)
        A, B = kroneckerfactortypes(A)
        return KroneckerAlgorithm(
            MAK.$f(A; kwargs..., kwargs1...),
            MAK.$f(B; kwargs..., kwargs2...)
        )
    end
end

for f in (
        :eig_full, :eigh_full,
        :left_polar, :right_polar,
        :lq_compact, :lq_full,
        :qr_compact, :qr_full,
        :svd_compact, :svd_full,
    )
    f! = Symbol(f, :!)
    @eval MAK.initialize_output(::typeof($f!), a::AbstractMatrix, alg::KroneckerAlgorithm) = nothing
    @eval MAK.$f!(a::AbstractKroneckerMatrix, F, alg::KroneckerAlgorithm) =
        otimes.(MAK.$f.(kroneckerfactors(a), kroneckerfactors(alg))...)
end

for f in (:eig_vals, :eigh_vals, :svd_vals)
    f! = Symbol(f, :!)
    @eval MAK.initialize_output(::typeof($f!), a::AbstractMatrix, alg::KroneckerAlgorithm) = nothing
    @eval function MAK.$f!(a::AbstractKroneckerMatrix, F, alg::KroneckerAlgorithm)
        d1 = MAK.$f(kroneckerfactors(a, 1), kroneckerfactors(alg, 1))
        d2 = MAK.$f(kroneckerfactors(a, 2), kroneckerfactors(alg, 2))
        return d1 ⊗ d2
    end
end

for f in (:left_orth, :right_orth)
    f! = Symbol(f, :!)
    @eval MAK.initialize_output(::typeof($f!), a::AbstractKroneckerMatrix) =
        nothing
    @eval function MAK.$f!(a::AbstractKroneckerMatrix, F; kwargs1 = (;), kwargs2 = (;), kwargs...)
        a1 = MAK.$f(kroneckerfactors(a, 1); kwargs..., kwargs1...)
        a2 = MAK.$f(kroneckerfactors(a, 2); kwargs..., kwargs2...)
        return a1 .⊗ a2
    end
end

for f in [:left_null, :right_null]
    f! = Symbol(f, :!)
    @eval MAK.initialize_output(::typeof($f!), a::AbstractKroneckerMatrix) =
        nothing
    @eval function MAK.$f!(a::AbstractKroneckerMatrix, F; kwargs1 = (;), kwargs2 = (;), kwargs...)
        a1 = MAK.$f(kroneckerfactors(a, 1); kwargs..., kwargs1...)
        a2 = MAK.$f(kroneckerfactors(a, 2); kwargs..., kwargs2...)
        return a1 ⊗ a2
    end
end

# Truncation


struct KroneckerTruncationStrategy{T <: TruncationStrategy} <: TruncationStrategy
    strategy::T
end

using FillArrays: OnesVector
const OnesKroneckerVector{T, A <: OnesVector{T}, B <: AbstractVector{T}} = KroneckerVector{T, A, B}
const KroneckerOnesVector{T, A <: AbstractVector{T}, B <: OnesVector{T}} = KroneckerVector{T, A, B}
const OnesVectorOnesVector{T, A <: OnesVector{T}, B <: OnesVector{T}} = KroneckerVector{T, A, B}

axis(a) = only(axes(a))

# Convert indices determined with a generic call to `findtruncated` to indices
# more suited for a KroneckerVector.
function to_truncated_indices(values::OnesKroneckerVector, I)
    prods = cartesianproduct(kroneckerfactors(axis(values))...)[I]
    I_id = only(to_indices(kroneckerfactors(values, 1), (:,)))
    I_data = unique(kroneckerfactors.(prods, 2))
    # Drop truncations that occur within the identity.
    I_data = filter(I_data) do i
        return count(x -> kroneckerfactors(x, 2) == i, prods) == length(kroneckerfactors(values, 2))
    end
    return I_id × I_data
end
function to_truncated_indices(values::KroneckerOnesVector, I)
    #I = findtruncated(Vector(values), strategy.strategy)
    prods = cartesianproduct(kroneckerfactors(axis(values))...)[I]
    I_data = unique(kroneckerfactors.(prods, 1))
    # Drop truncations that occur within the identity.
    I_data = filter(I_data) do i
        return count(x -> kroneckerfactors(x, 1) == i, prods) == length(kroneckerfactors(values, 2))
    end
    I_id = only(to_indices(kroneckerfactors(values, 2), (:,)))
    return I_data × I_id
end
# Fix ambiguity error.
function to_truncated_indices(values::OnesVectorOnesVector, I)
    return throw(ArgumentError("Not implemented"))
end
function to_truncated_indices(values::KroneckerVector, I)
    return throw(ArgumentError("Not implemented"))
end

function MAK.findtruncated(
        values::AbstractKroneckerVector, strategy::KroneckerTruncationStrategy
    )
    I = findtruncated(Vector(values), strategy.strategy)
    return to_truncated_indices(values, I)
end

for f in (:eig_trunc!, :eigh_trunc!)
    @eval function MAK.truncate(
            ::typeof($f), DV::NTuple{2, AbstractKroneckerMatrix}, strategy::TruncationStrategy
        )
        return MAK.truncate($f, DV, KroneckerTruncationStrategy(strategy))
    end
    @eval function MAK.truncate(
            ::typeof($f), (D, V)::NTuple{2, AbstractKroneckerMatrix}, strategy::KroneckerTruncationStrategy
        )
        I = MAK.findtruncated(MAK.diagview(D), strategy)
        return (D[I, I], V[(:) × (:), I]), I
    end
end

MAK.truncate(f::typeof(svd_trunc!), USVᴴ::NTuple{3, AbstractKroneckerMatrix}, strategy::TruncationStrategy) =
    MAK.truncate(f, USVᴴ, KroneckerTruncationStrategy(strategy))
function MAK.truncate(
        ::typeof(svd_trunc!), (U, S, Vᴴ)::NTuple{3, AbstractKroneckerMatrix}, strategy::KroneckerTruncationStrategy,
    )
    I = MAK.findtruncated(MAK.diagview(S), strategy)
    return (U[(:) × (:), I], S[I, I], Vᴴ[I, (:) × (:)]), I
end
