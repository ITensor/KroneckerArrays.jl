module KroneckerArraysTensorAlgebraExt

using KroneckerArrays: KroneckerArrays, AbstractKroneckerArray, ⊗, kroneckerfactors
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTrivialPermutation,
    FusionStyle, matricize, unmatricize

struct KroneckerFusion{A <: FusionStyle, B <: FusionStyle} <: FusionStyle
    a::A
    b::B
end
KroneckerArrays.kroneckerfactors(style::KroneckerFusion) = (style.a, style.b)
KroneckerArrays.kroneckerfactortypes(::Type{KroneckerFusion{A, B}}) where {A, B} = (A, B)

function TensorAlgebra.FusionStyle(a::AbstractKroneckerArray)
    return KroneckerFusion(FusionStyle.(kroneckerfactors(a))...)
end
function matricize_kronecker(
        style::FusionStyle, a::AbstractArray, length1::Val, length2::Val
    )
    m1 = matricize(kroneckerfactors(style, 1), kroneckerfactors(a, 1), length1, length2)
    m2 = matricize(kroneckerfactors(style, 2), kroneckerfactors(a, 2), length1, length2)
    return m1 ⊗ m2
end
function TensorAlgebra.matricize(
        style::KroneckerFusion, a::AbstractArray, length1::Val, length2::Val
    )
    return matricize_kronecker(style, a, length1, length2)
end
function unmatricize_kronecker(
        style::FusionStyle,
        m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    style1, style2 = kroneckerfactors(style)
    m1, m2 = kroneckerfactors(m)
    codomain1 = kroneckerfactors.(codomain_axes, 1)
    codomain2 = kroneckerfactors.(codomain_axes, 2)
    domain1 = kroneckerfactors.(domain_axes, 1)
    domain2 = kroneckerfactors.(domain_axes, 2)
    a1 = unmatricize(style1, m1, codomain1, domain1)
    a2 = unmatricize(style2, m2, codomain2, domain2)
    return a1 ⊗ a2
end
function TensorAlgebra.unmatricize(
        style::KroneckerFusion,
        m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return unmatricize_kronecker(style, m, codomain_axes, domain_axes)
end

end
