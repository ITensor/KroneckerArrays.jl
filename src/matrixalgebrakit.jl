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
    eig_full,
    eig_trunc!,
    eig_trunc,
    eig_vals!,
    eig_vals,
    eigh_full!,
    eigh_full,
    eigh_trunc!,
    eigh_trunc,
    eigh_vals!,
    eigh_vals,
    initialize_output,
    left_null!,
    left_null,
    left_orth!,
    left_orth,
    left_polar!,
    left_polar,
    lq_compact!,
    lq_compact,
    lq_full!,
    lq_full,
    qr_compact!,
    qr_compact,
    qr_full!,
    qr_full,
    right_null!,
    right_null,
    right_orth!,
    right_orth,
    right_polar!,
    right_polar,
    svd_compact!,
    svd_compact,
    svd_full!,
    svd_full,
    svd_trunc!,
    svd_trunc,
    svd_vals!,
    svd_vals,
    truncate

using DiagonalArrays: DiagonalArrays, diagview
function DiagonalArrays.diagview(a::KroneckerMatrix)
    return diagview(arg1(a)) ⊗ diagview(arg2(a))
end
MatrixAlgebraKit.diagview(a::KroneckerMatrix) = diagview(a)

struct KroneckerAlgorithm{A1, A2} <: AbstractAlgorithm
    arg1::A1
    arg2::A2
end
@inline arg1(alg::KroneckerAlgorithm) = getfield(alg, :arg1)
@inline arg2(alg::KroneckerAlgorithm) = getfield(alg, :arg2)

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
            return copy_input($f, arg1(a)) ⊗ copy_input($f, arg2(a))
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
    @eval begin
        function MatrixAlgebraKit.$f(
                A::Type{<:KroneckerMatrix}; kwargs1 = (;), kwargs2 = (;), kwargs...
            )
            A1, A2 = argument_types(A)
            return KroneckerAlgorithm(
                $f(A1; kwargs..., kwargs1...), $f(A2; kwargs..., kwargs2...)
            )
        end
    end
end

for f in [
        :eig_full,
        :eigh_full,
        :left_polar,
        :lq_compact,
        :lq_full,
        :qr_compact,
        :qr_full,
        :right_polar,
        :svd_compact,
        :svd_full,
    ]
    f! = Symbol(f, :!)
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof($f!), a::AbstractMatrix, alg::KroneckerAlgorithm
            )
            return nothing
        end
        function MatrixAlgebraKit.$f!(
                a::KroneckerMatrix, F, alg::KroneckerAlgorithm; kwargs1 = (;), kwargs2 = (;), kwargs...
            )
            a1 = $f(arg1(a), arg1(alg); kwargs..., kwargs1...)
            a2 = $f(arg2(a), arg2(alg); kwargs..., kwargs2...)
            return a1 .⊗ a2
        end
    end
end

for f in [:eig_vals, :eigh_vals, :svd_vals]
    f! = Symbol(f, :!)
    @eval begin
        function MatrixAlgebraKit.initialize_output(
                ::typeof($f!), a::AbstractMatrix, alg::KroneckerAlgorithm
            )
            return nothing
        end
        function MatrixAlgebraKit.$f!(
                a::KroneckerMatrix, F, alg::KroneckerAlgorithm; kwargs1 = (;), kwargs2 = (;), kwargs...
            )
            a1 = $f(arg1(a), arg1(alg); kwargs..., kwargs1...)
            a2 = $f(arg2(a), arg2(alg); kwargs..., kwargs2...)
            return a1 ⊗ a2
        end
    end
end

for f in [:left_orth, :right_orth]
    f! = Symbol(f, :!)
    @eval begin
        function MatrixAlgebraKit.initialize_output(::typeof($f!), a::KroneckerMatrix)
            return nothing
        end
        function MatrixAlgebraKit.$f!(
                a::KroneckerMatrix, F; kwargs1 = (;), kwargs2 = (;), kwargs...
            )
            a1 = $f(arg1(a); kwargs..., kwargs1...)
            a2 = $f(arg2(a); kwargs..., kwargs2...)
            return a1 .⊗ a2
        end
    end
end

for f in [:left_null, :right_null]
    f! = Symbol(f, :!)
    @eval begin
        function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
            return nothing
        end
        function MatrixAlgebraKit.$f!(
                a::KroneckerMatrix, F; kwargs1 = (;), kwargs2 = (;), kwargs...
            )
            a1 = $f(arg1(a); kwargs..., kwargs1...)
            a2 = $f(arg2(a); kwargs..., kwargs2...)
            return a1 ⊗ a2
        end
    end
end

# Truncation

using MatrixAlgebraKit: TruncationStrategy, findtruncated, truncate

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
    prods = cartesianproduct(axis(values))[I]
    I_id = only(to_indices(arg1(values), (:,)))
    I_data = unique(arg2.(prods))
    # Drop truncations that occur within the identity.
    I_data = filter(I_data) do i
        return count(x -> arg2(x) == i, prods) == length(arg2(values))
    end
    return I_id × I_data
end
function to_truncated_indices(values::KroneckerOnesVector, I)
    #I = findtruncated(Vector(values), strategy.strategy)
    prods = cartesianproduct(axis(values))[I]
    I_data = unique(arg1.(prods))
    # Drop truncations that occur within the identity.
    I_data = filter(I_data) do i
        return count(x -> arg1(x) == i, prods) == length(arg2(values))
    end
    I_id = only(to_indices(arg2(values), (:,)))
    return I_data × I_id
end
# Fix ambiguity error.
function to_truncated_indices(values::OnesVectorOnesVector, I)
    return throw(ArgumentError("Not implemented"))
end
function to_truncated_indices(values::KroneckerVector, I)
    return throw(ArgumentError("Not implemented"))
end

function MatrixAlgebraKit.findtruncated(
        values::KroneckerVector, strategy::KroneckerTruncationStrategy
    )
    I = findtruncated(Vector(values), strategy.strategy)
    return to_truncated_indices(values, I)
end

for f in [:eig_trunc!, :eigh_trunc!]
    @eval begin
        function MatrixAlgebraKit.truncate(
                ::typeof($f), DV::NTuple{2, KroneckerMatrix}, strategy::TruncationStrategy
            )
            return truncate($f, DV, KroneckerTruncationStrategy(strategy))
        end
        function MatrixAlgebraKit.truncate(
                ::typeof($f), (D, V)::NTuple{2, KroneckerMatrix}, strategy::KroneckerTruncationStrategy
            )
            I = findtruncated(diagview(D), strategy)
            return (D[I, I], V[(:) × (:), I]), I
        end
    end
end

function MatrixAlgebraKit.truncate(
        f::typeof(svd_trunc!), USVᴴ::NTuple{3, KroneckerMatrix}, strategy::TruncationStrategy
    )
    return truncate(f, USVᴴ, KroneckerTruncationStrategy(strategy))
end
function MatrixAlgebraKit.truncate(
        ::typeof(svd_trunc!),
        (U, S, Vᴴ)::NTuple{3, KroneckerMatrix},
        strategy::KroneckerTruncationStrategy,
    )
    I = findtruncated(diagview(S), strategy)
    return (U[(:) × (:), I], S[I, I], Vᴴ[I, (:) × (:)]), I
end
