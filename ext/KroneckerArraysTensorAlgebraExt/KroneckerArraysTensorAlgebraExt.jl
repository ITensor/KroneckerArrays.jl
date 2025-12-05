module KroneckerArraysTensorAlgebraExt

using KroneckerArrays: KroneckerArrays, AbstractKroneckerArray, CartesianProductUnitRange,
    ⊗, cartesianrange, kroneckerfactors, kroneckerfactortypes
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTrivialPermutation,
    FusionStyle, matricize, tensor_product_axis, unmatricize

struct KroneckerFusion{A <: FusionStyle, B <: FusionStyle} <: FusionStyle
    a::A
    b::B
end
KroneckerArrays.kroneckerfactors(style::KroneckerFusion) = (style.a, style.b)
KroneckerArrays.kroneckerfactortypes(::Type{KroneckerFusion{A, B}}) where {A, B} = (A, B)

function TensorAlgebra.FusionStyle(A::Type{<:AbstractKroneckerArray})
    return KroneckerFusion(FusionStyle.(kroneckerfactortypes(A))...)
end
function TensorAlgebra.FusionStyle(A::Type{<:CartesianProductUnitRange})
    return KroneckerFusion(FusionStyle.(kroneckerfactortypes(A))...)
end

function TensorAlgebra.tensor_product_axis(
        style::KroneckerFusion, r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    style_a, style_b = kroneckerfactors(style)
    r1a, r1b = kroneckerfactors(r1)
    r2a, r2b = kroneckerfactors(r2)
    ra = tensor_product_axis(style_a, r1a, r2a)
    rb = tensor_product_axis(style_b, r1b, r2b)
    return cartesianrange(ra, rb)
end

function matricize_kronecker(
        style::FusionStyle, a::AbstractArray, length_codomain::Val
    )
    m1 = matricize(kroneckerfactors(style, 1), kroneckerfactors(a, 1), length_codomain)
    m2 = matricize(kroneckerfactors(style, 2), kroneckerfactors(a, 2), length_codomain)
    return m1 ⊗ m2
end
function TensorAlgebra.matricize(
        style::KroneckerFusion, a::AbstractArray, length_codomain::Val
    )
    return matricize_kronecker(style, a, length_codomain)
end

function unmatricize_kronecker(
        style::FusionStyle,
        m::AbstractMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    style1, style2 = kroneckerfactors(style)
    m1, m2 = kroneckerfactors(m)
    codomain1 = kroneckerfactors.(axes_codomain, 1)
    codomain2 = kroneckerfactors.(axes_codomain, 2)
    domain1 = kroneckerfactors.(axes_domain, 1)
    domain2 = kroneckerfactors.(axes_domain, 2)
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
