module KroneckerArraysTensorAlgebraExt

using KroneckerArrays: KroneckerArrays, AbstractKroneckerArray, CartesianProductUnitRange,
    ⊗, cartesianrange, kroneckerfactors, kroneckerfactortypes
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTrivialPermutation,
    FusionStyle, matricize, tensor_product_axis, trivial_axis, unmatricize

struct KroneckerFusion{A <: FusionStyle, B <: FusionStyle} <: FusionStyle
    a::A
    b::B
end
function KroneckerFusion{A, B}() where {A <: FusionStyle, B <: FusionStyle}
    return KroneckerFusion{A, B}(A(), B())
end
KroneckerArrays.kroneckerfactors(style::KroneckerFusion) = (style.a, style.b)
KroneckerArrays.kroneckerfactortypes(::Type{KroneckerFusion{A, B}}) where {A, B} = (A, B)
function KroneckerArrays.:⊗(style1::FusionStyle, style2::FusionStyle)
    return KroneckerFusion(style1, style2)
end

function TensorAlgebra.FusionStyle(A::Type{<:AbstractKroneckerArray})
    return KroneckerFusion(FusionStyle.(kroneckerfactortypes(A))...)
end
function TensorAlgebra.FusionStyle(A::Type{<:CartesianProductUnitRange})
    return KroneckerFusion(FusionStyle.(kroneckerfactortypes(A))...)
end

function TensorAlgebra.trivial_axis(
        style::KroneckerFusion, side::Val{:codomain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return trivial_kronecker(style, side, a, axes_codomain, axes_domain)
end
function TensorAlgebra.trivial_axis(
        style::KroneckerFusion, side::Val{:domain}, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    return trivial_kronecker(style, side, a, axes_codomain, axes_domain)
end
function trivial_kronecker(
        style::FusionStyle, side::Val, a::AbstractArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}},
    )
    style_a, style_b = kroneckerfactors(style)
    a_a, a_b = kroneckerfactors(a)
    axes_codomain_a = kroneckerfactors.(axes_codomain, 1)
    axes_codomain_b = kroneckerfactors.(axes_codomain, 2)
    axes_domain_a = kroneckerfactors.(axes_domain, 1)
    axes_domain_b = kroneckerfactors.(axes_domain, 2)
    ra = trivial_axis(style_a, side, a_a, axes_codomain_a, axes_domain_a)
    rb = trivial_axis(style_b, side, a_b, axes_codomain_b, axes_domain_b)
    return cartesianrange(ra, rb)
end

function TensorAlgebra.tensor_product_axis(
        style::KroneckerFusion, side::Val{:codomain},
        r1::AbstractUnitRange, r2::AbstractUnitRange,
    )
    return tensor_product_kronecker(style, side, r1, r2)
end
function TensorAlgebra.tensor_product_axis(
        style::KroneckerFusion, side::Val{:domain},
        r1::AbstractUnitRange, r2::AbstractUnitRange,
    )
    return tensor_product_kronecker(style, side, r1, r2)
end
function tensor_product_kronecker(
        style::KroneckerFusion, side::Val,
        r1::AbstractUnitRange, r2::AbstractUnitRange,
    )
    style_a, style_b = kroneckerfactors(style)
    r1a, r1b = kroneckerfactors(r1)
    r2a, r2b = kroneckerfactors(r2)
    ra = tensor_product_axis(style_a, side, r1a, r2a)
    rb = tensor_product_axis(style_b, side, r1b, r2b)
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
        style::FusionStyle, m::AbstractMatrix,
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
        style::KroneckerFusion, m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return unmatricize_kronecker(style, m, codomain_axes, domain_axes)
end

end
