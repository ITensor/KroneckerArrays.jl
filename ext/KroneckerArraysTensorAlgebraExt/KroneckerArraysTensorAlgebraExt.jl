module KroneckerArraysTensorAlgebraExt

using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗, arg1, arg2
using TensorAlgebra:
    TensorAlgebra, AbstractBlockPermutation, FusionStyle, matricize, unmatricize

struct KroneckerFusion{A <: FusionStyle, B <: FusionStyle} <: FusionStyle
    a::A
    b::B
end
KroneckerArrays.arg1(style::KroneckerFusion) = style.a
KroneckerArrays.arg2(style::KroneckerFusion) = style.b
function TensorAlgebra.FusionStyle(a::KroneckerArray)
    return KroneckerFusion(FusionStyle(arg1(a)), FusionStyle(arg2(a)))
end
function matricize_kronecker(
        style::KroneckerFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
    )
    return matricize(arg1(style), arg1(a), biperm) ⊗ matricize(arg2(style), arg2(a), biperm)
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
    return unmatricize(arg1(style), arg1(a), arg1.(ax)) ⊗
        unmatricize(arg2(style), arg2(a), arg2.(ax))
end
function TensorAlgebra.unmatricize(style::KroneckerFusion, a::AbstractArray, ax)
    return unmatricize_kronecker(style, a, ax)
end

end
