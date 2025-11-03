using DiagonalArrays: δ
using LinearAlgebra:
    LinearAlgebra,
    Diagonal,
    Eigen,
    SVD,
    det,
    diag,
    eigen,
    eigvals,
    lq,
    mul!,
    norm,
    qr,
    svd,
    svdvals,
    tr

using LinearAlgebra: LinearAlgebra
function KroneckerArray(J::LinearAlgebra.UniformScaling, ax::Tuple)
    return δ(eltype(J), arg1.(ax)) ⊗ δ(eltype(J), arg2.(ax))
end
function Base.copyto!(a::KroneckerArray, J::LinearAlgebra.UniformScaling)
    copyto!(a, KroneckerArray(J, axes(a)))
    return a
end

using LinearAlgebra: LinearAlgebra, pinv
function LinearAlgebra.pinv(a::KroneckerArray; kwargs...)
    return pinv(arg1(a); kwargs...) ⊗ pinv(arg2(a); kwargs...)
end

function LinearAlgebra.diag(a::AbstractKroneckerArray)
    return copy(DiagonalArrays.diagview(a))
end

function Base.:*(a::AbstractKroneckerArray, b::AbstractKroneckerArray)
    return (arg1(a) * arg1(b)) ⊗ (arg2(a) * arg2(b))
end

function LinearAlgebra.mul!(
        c::AbstractKroneckerArray, a::AbstractKroneckerArray, b::AbstractKroneckerArray, α::Number, β::Number
    )
    iszero(β) || iszero(c) || throw(
        ArgumentError(
            "Can't multiply KroneckerArrays with nonzero β and nonzero destination."
        ),
    )
    # TODO: Only perform in-place operation on the non-active argument(s).
    mul!(arg1(c), arg1(a), arg1(b))
    mul!(arg2(c), arg2(a), arg2(b), α, β)
    return c
end

using LinearAlgebra: tr
function LinearAlgebra.tr(a::AbstractKroneckerArray)
    return tr(arg1(a)) * tr(arg2(a))
end

using LinearAlgebra: norm
function LinearAlgebra.norm(a::AbstractKroneckerArray, p::Int = 2)
    return norm(arg1(a), p) * norm(arg2(a), p)
end

# Matrix functions
const MATRIX_FUNCTIONS = [
    :exp,
    :cis,
    :log,
    :sqrt,
    :cbrt,
    :cos,
    :sin,
    :tan,
    :csc,
    :sec,
    :cot,
    :cosh,
    :sinh,
    :tanh,
    :csch,
    :sech,
    :coth,
    :acos,
    :asin,
    :atan,
    :acsc,
    :asec,
    :acot,
    :acosh,
    :asinh,
    :atanh,
    :acsch,
    :asech,
    :acoth,
]

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(a::AbstractKroneckerArray)
            return if isone(arg1(a))
                arg1(a) ⊗ $f(arg2(a))
            elseif isone(arg2(a))
                $f(arg1(a)) ⊗ arg2(a)
            else
                throw(ArgumentError("Generic KroneckerArray `$($f)` is not supported."))
            end
        end
    end
end

# `DiagonalArrays.issquare` and `DiagonalArrays.checksquare` are more general
# than `LinearAlgebra.checksquare`, for example it compares axes and can check
# that the codomain and domain are dual of each other.
using DiagonalArrays: DiagonalArrays, checksquare, issquare
function DiagonalArrays.issquare(a::AbstractKroneckerArray)
    return issquare(arg1(a)) && issquare(arg2(a))
end

using LinearAlgebra: det
function LinearAlgebra.det(a::AbstractKroneckerArray)
    checksquare(a)
    return det(arg1(a))^size(arg2(a), 1) * det(arg2(a))^size(arg1(a), 1)
end

function LinearAlgebra.svd(a::AbstractKroneckerArray)
    F1 = svd(arg1(a))
    F2 = svd(arg2(a))
    return SVD(F1.U ⊗ F2.U, F1.S ⊗ F2.S, F1.Vt ⊗ F2.Vt)
end
function LinearAlgebra.svdvals(a::AbstractKroneckerArray)
    return svdvals(arg1(a)) ⊗ svdvals(arg2(a))
end
function LinearAlgebra.eigen(a::AbstractKroneckerArray)
    F1 = eigen(arg1(a))
    F2 = eigen(arg2(a))
    return Eigen(F1.values ⊗ F2.values, F1.vectors ⊗ F2.vectors)
end
function LinearAlgebra.eigvals(a::AbstractKroneckerArray)
    return eigvals(arg1(a)) ⊗ eigvals(arg2(a))
end

struct KroneckerQ{A1, A2}
    arg1::A1
    arg2::A2
end
@inline arg1(a::KroneckerQ) = getfield(a, :arg1)
@inline arg2(a::KroneckerQ) = getfield(a, :arg2)
function Base.:*(a::KroneckerQ, b::KroneckerQ)
    return (arg1(a) * arg1(b)) ⊗ (arg2(a) * arg2(b))
end
function Base.:*(a1::KroneckerQ, a2::AbstractKroneckerArray)
    return (arg1(a1) * arg1(a2)) ⊗ (arg2(a1) * arg2(a2))
end
function Base.:*(a1::AbstractKroneckerArray, a2::KroneckerQ)
    return (arg1(a1) * arg1(a2)) ⊗ (arg2(a1) * arg2(a2))
end
function Base.adjoint(a::KroneckerQ)
    return KroneckerQ(arg1(a)', arg2(a)')
end

struct KroneckerQR{QQ, RR}
    Q::QQ
    R::RR
end
Base.iterate(F::KroneckerQR) = (F.Q, Val(:R))
Base.iterate(F::KroneckerQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::KroneckerQR, ::Val{:done}) = nothing
function ⊗(a1::LinearAlgebra.QRCompactWYQ, a2::LinearAlgebra.QRCompactWYQ)
    return KroneckerQ(a1, a2)
end
function LinearAlgebra.qr(a::AbstractKroneckerArray)
    Fa = qr(arg1(a))
    Fb = qr(arg2(a))
    return KroneckerQR(Fa.Q ⊗ Fb.Q, Fa.R ⊗ Fb.R)
end

struct KroneckerLQ{LL, QQ}
    L::LL
    Q::QQ
end
Base.iterate(F::KroneckerLQ) = (F.L, Val(:Q))
Base.iterate(F::KroneckerLQ, ::Val{:Q}) = (F.Q, Val(:done))
Base.iterate(F::KroneckerLQ, ::Val{:done}) = nothing
function ⊗(a1::LinearAlgebra.LQPackedQ, a2::LinearAlgebra.LQPackedQ)
    return KroneckerQ(a1, a2)
end
function LinearAlgebra.lq(a::AbstractKroneckerArray)
    Fa = lq(arg1(a))
    Fb = lq(arg2(a))
    return KroneckerLQ(Fa.L ⊗ Fb.L, Fa.Q ⊗ Fb.Q)
end
