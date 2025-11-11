KroneckerArray(J::LinearAlgebra.UniformScaling, ax::Tuple) =
    DiagonalArrays.δ(eltype(J), kroneckerfactors.(ax, 1)) ⊗ DiagonalArrays.δ(eltype(J), kroneckerfactors.(ax, 2))

function Base.copyto!(a::KroneckerArray, J::LinearAlgebra.UniformScaling)
    copyto!(a, KroneckerArray(J, axes(a)))
    return a
end

LinearAlgebra.pinv(a::KroneckerArray; kwargs...) = ⊗(LinearAlgebra.pinv.(kroneckerfactors(a); kwargs...)...)
LinearAlgebra.diag(a::AbstractKroneckerArray) = copy(DiagonalArrays.diagview(a))
LinearAlgebra.tr(a::AbstractKroneckerArray) = *(LinearAlgebra.tr.(kroneckerfactors(a))...)
LinearAlgebra.norm(a::AbstractKroneckerArray, p::Real = 2) = *(LinearAlgebra.norm.(kroneckerfactors(a), p)...)

function Base.:*(A::AbstractKroneckerArray, B::AbstractKroneckerArray)
    a, b = kroneckerfactors(A)
    c, d = kroneckerfactors(B)
    return (a * c) ⊗ (b * d)
end

function LinearAlgebra.mul!(
        c::AbstractKroneckerArray, a::AbstractKroneckerArray, b::AbstractKroneckerArray, α::Number, β::Number
    )
    iszero(β) || iszero(c) ||
        throw(ArgumentError("Can't multiply KroneckerArrays with nonzero β and nonzero destination."))
    # TODO: Only perform in-place operation on the non-active argument(s).
    ca, cb = kroneckerfactors(c)
    aa, ab = kroneckerfactors(a)
    ba, bb = kroneckerfactors(b)
    LinearAlgebra.mul!(ca, aa, ba)
    LinearAlgebra.mul!(cb, ab, bb, α, β)
    return c
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
    @eval function Base.$f(ab::AbstractKroneckerArray)
        a, b = kroneckerfactors(ab)
        return if isone(a)
            a ⊗ $f(b)
        elseif isone(b)
            $f(a) ⊗ b
        else
            throw(ArgumentError("Generic KroneckerArray `$($f)` is not supported."))
        end
    end
end

# `DiagonalArrays.issquare` and `DiagonalArrays.checksquare` are more general
# than `LinearAlgebra.checksquare`, for example it compares axes and can check
# that the codomain and domain are dual of each other.
using DiagonalArrays: DiagonalArrays, checksquare, issquare
function DiagonalArrays.issquare(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    return DiagonalArrays.issquare(a) && DiagonalArrays.issquare(b)
end

function LinearAlgebra.det(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    return LinearAlgebra.det(a)^size(b, 1) * LinearAlgebra.det(b)^size(a, 1)
end

function LinearAlgebra.svd(ab::AbstractKroneckerArray; kwargs...)
    Fa, Fb = LinearAlgebra.svd.(kroneckerfactors(ab); kwargs...)
    return LinearAlgebra.SVD(Fa.U ⊗ Fb.U, Fa.S ⊗ Fb.S, Fa.Vt ⊗ Fb.Vt)
end
LinearAlgebra.svdvals(a::AbstractKroneckerArray) = ⊗(LinearAlgebra.svdvals.(kroneckerfactors(a))...)

function LinearAlgebra.eigen(a::AbstractKroneckerArray; kwargs...)
    Fa, Fb = LinearAlgebra.eigen.(kroneckerfactors(a); kwargs...)
    return LinearAlgebra.Eigen(Fa.values ⊗ Fb.values, Fa.vectors ⊗ Fb.vectors)
end
LinearAlgebra.eigvals(a::AbstractKroneckerArray) = ⊗(LinearAlgebra.eigvals.(kroneckerfactors(a))...)

struct KroneckerQ{A, B}
    a::A
    b::B
end
kroneckerfactors(q::KroneckerQ) = q.a, q.b
kroneckerfactortypes(::Type{KroneckerQ{A, B}}) where {A, B} = (A, B)

Base.:*(a::KroneckerQ, b::KroneckerQ) = ⊗((kroneckerfactors(a) .* kroneckerfactors(b))...)
Base.:*(a::KroneckerQ, b::AbstractKroneckerArray) = ⊗((kroneckerfactors(a) .* kroneckerfactors(b))...)
Base.:*(a::AbstractKroneckerArray, b::KroneckerQ) = ⊗((kroneckerfactors(a) .* kroneckerfactors(b))...)
Base.adjoint(a::KroneckerQ) = KroneckerQ(adjoint.(kroneckerfactors(a))...)

struct KroneckerQR{QQ, RR}
    Q::QQ
    R::RR
end
Base.iterate(F::KroneckerQR) = (F.Q, Val(:R))
Base.iterate(F::KroneckerQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::KroneckerQR, ::Val{:done}) = nothing

function ⊗(a::LinearAlgebra.QRCompactWYQ, b::LinearAlgebra.QRCompactWYQ)
    return KroneckerQ(a, b)
end

function LinearAlgebra.qr(a::AbstractKroneckerArray)
    Fa, Fb = LinearAlgebra.qr.(kroneckerfactors(a))
    return KroneckerQR(Fa.Q ⊗ Fb.Q, Fa.R ⊗ Fb.R)
end

struct KroneckerLQ{LL, QQ}
    L::LL
    Q::QQ
end
Base.iterate(F::KroneckerLQ) = (F.L, Val(:Q))
Base.iterate(F::KroneckerLQ, ::Val{:Q}) = (F.Q, Val(:done))
Base.iterate(F::KroneckerLQ, ::Val{:done}) = nothing

function ⊗(a::LinearAlgebra.LQPackedQ, b::LinearAlgebra.LQPackedQ)
    return KroneckerQ(a, b)
end

function LinearAlgebra.lq(a::AbstractKroneckerArray)
    Fa, Fb = LinearAlgebra.lq.(kroneckerfactors(a))
    return KroneckerLQ(Fa.L ⊗ Fb.L, Fa.Q ⊗ Fb.Q)
end
