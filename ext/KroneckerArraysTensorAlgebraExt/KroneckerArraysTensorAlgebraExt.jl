module KroneckerArraysTensorAlgebraExt

using KroneckerArrays: KroneckerArrays, AbstractKroneckerArray, ⊗, kroneckerfactors
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTrivialPermutation, FusionStyle,
    matricize, unmatricize

struct KroneckerFusion{A <: FusionStyle, B <: FusionStyle} <: FusionStyle
    a::A
    b::B
end
KroneckerArrays.kroneckerfactors(style::KroneckerFusion) = (style.a, style.b)
KroneckerArrays.kroneckerfactortypes(::Type{KroneckerFusion{A, B}}) where {A, B} = (A, B)

TensorAlgebra.FusionStyle(a::AbstractKroneckerArray) = KroneckerFusion(FusionStyle.(kroneckerfactors(a))...)
function matricize_kronecker(
        style::KroneckerFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
    )
    return matricize(kroneckerfactors(style, 1), kroneckerfactors(a, 1), biperm) ⊗
        matricize(kroneckerfactors(style, 2), kroneckerfactors(a, 2), biperm)
end
function TensorAlgebra.matricize(
        style::KroneckerFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
    )
    return matricize_kronecker(style, a, biperm)
end
# Fix ambiguity error.
# TODO: Investigate rewriting the logic in `TensorAlgebra.jl` to avoid this.
using TensorAlgebra: BlockedTrivialPermutation, unmatricize
function TensorAlgebra.matricize(
        style::KroneckerFusion, a::AbstractArray, biperm::BlockedTrivialPermutation{2}
    )
    return matricize_kronecker(style, a, biperm)
end
function unmatricize_kronecker(style::KroneckerFusion, a::AbstractArray, ax)
    return unmatricize(kroneckerfactors(style, 1), kroneckerfactors(a, 1), kroneckerfactors.(ax, 1)) ⊗
        unmatricize(kroneckerfactors(style, 2), kroneckerfactors(a, 2), kroneckerfactors.(ax, 2))
end
function TensorAlgebra.unmatricize(style::KroneckerFusion, a::AbstractArray, ax)
    return unmatricize_kronecker(style, a, ax)
end

end
