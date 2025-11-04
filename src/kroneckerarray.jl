"""
    abstract type AbstractKroneckerArray{T, N} <: AbstractArray{T, N} end

Abstract supertype for arrays that have a kronecker product structure,
i.e. that can be written as `AB = A ⊗ B`.
"""
abstract type AbstractKroneckerArray{T, N} <: AbstractArray{T, N} end

const AbstractKroneckerVector{T} = AbstractKroneckerArray{T, 1}
const AbstractKroneckerMatrix{T} = AbstractKroneckerArray{T, 2}

@doc """
    arg1(AB::AbstractKroneckerArray{T, N})

Extract the first factor (`A`) of the Kronecker array `AB = A ⊗ B`.
""" arg1

@doc """
    arg2(AB::AbstractKroneckerArray{T, N})

Extract the second factor (`B`) of the Kronecker array `AB = A ⊗ B`.
""" arg2

arg1type(x::AbstractKroneckerArray) = arg1type(typeof(x))
arg1type(::Type{<:AbstractKroneckerArray}) = error("`AbstractKroneckerArray` subtypes have to implement `arg1type`.")
arg2type(x::AbstractKroneckerArray) = arg2type(typeof(x))
arg2type(::Type{<:AbstractKroneckerArray}) = error("`AbstractKroneckerArray` subtypes have to implement `arg2type`.")

arguments(a::AbstractKroneckerArray) = (arg1(a), arg2(a))
arguments(a::AbstractKroneckerArray, n::Int) = arguments(a)[n]
argument_types(a::AbstractKroneckerArray) = argument_types(typeof(a))

function unwrap_array(a::AbstractArray)
    p = parent(a)
    p ≡ a && return a
    return unwrap_array(p)
end
isactive(a::AbstractArray) = ismutable(unwrap_array(a))

using TypeParameterAccessors: unwrap_array_type
function isactive(arrayt::Type{<:AbstractArray})
    return ismutabletype(unwrap_array_type(arrayt))
end

# Custom `_convert` works around the issue that
# `convert(::Type{<:Diagonal}, ::AbstractMatrix)` isn't defined
# in Julia v1.10 (https://github.com/JuliaLang/julia/pull/48895,
# https://github.com/JuliaLang/julia/pull/52487).
# TODO: Delete once we drop support for Julia v1.10.
function _convert(A::Type{<:AbstractArray}, a::AbstractArray)
    return convert(A, a)
end
using LinearAlgebra: LinearAlgebra, Diagonal, diag, isdiag
_construct(A::Type{<:Diagonal}, a::AbstractMatrix) = A(diag(a))
function _convert(A::Type{<:Diagonal}, a::AbstractMatrix)
    LinearAlgebra.checksquare(a)
    return isdiag(a) ? _construct(A, a) : throw(InexactError(:convert, A, a))
end

struct KroneckerArray{T, N, A1 <: AbstractArray{T, N}, A2 <: AbstractArray{T, N}} <:
    AbstractKroneckerArray{T, N}
    arg1::A1
    arg2::A2
end
function KroneckerArray(a1::AbstractArray, a2::AbstractArray)
    if ndims(a1) != ndims(a2)
        throw(
            ArgumentError("Kronecker product requires arrays of the same number of dimensions.")
        )
    end
    elt = promote_type(eltype(a1), eltype(a2))
    return _convert(AbstractArray{elt}, a1) ⊗ _convert(AbstractArray{elt}, a2)
end
const KroneckerMatrix{T, A1 <: AbstractMatrix{T}, A2 <: AbstractMatrix{T}} = KroneckerArray{
    T, 2, A1, A2,
}
const KroneckerVector{T, A1 <: AbstractVector{T}, A2 <: AbstractVector{T}} = KroneckerArray{
    T, 1, A1, A2,
}

@inline arg1(a::KroneckerArray) = getfield(a, :arg1)
@inline arg2(a::KroneckerArray) = getfield(a, :arg2)
arg1type(::Type{KroneckerArray{T, N, A1, A2}}) where {T, N, A1, A2} = A1
arg2type(::Type{KroneckerArray{T, N, A1, A2}}) where {T, N, A1, A2} = A2

argument_types(::Type{<:KroneckerArray{<:Any, <:Any, A1, A2}}) where {A1, A2} = (A1, A2)

function mutate_active_args!(f!, f, dest, src)
    (isactive(arg1(dest)) || isactive(arg2(dest))) ||
        error("Can't mutate immutable KroneckerArray.")
    if isactive(arg1(dest))
        f!(arg1(dest), arg1(src))
    else
        arg1(dest) == f(arg1(src)) || error("Immutable arguments aren't equal.")
    end
    if isactive(arg2(dest))
        f!(arg2(dest), arg2(src))
    else
        arg2(dest) == f(arg2(src)) || error("Immutable arguments aren't equal.")
    end
    return dest
end

using Adapt: Adapt, adapt
function Adapt.adapt_structure(to, a::AbstractKroneckerArray)
    # TODO: Is this a good definition? It is similar to
    # the definition of `similar`.
    return if isactive(arg1(a)) == isactive(arg2(a))
        adapt(to, arg1(a)) ⊗ adapt(to, arg2(a))
    elseif isactive(arg1(a))
        adapt(to, arg1(a)) ⊗ arg2(a)
    elseif isactive(arg2(a))
        arg1(a) ⊗ adapt(to, arg2(a))
    end
end

Base.copy(a::AbstractKroneckerArray) = copy(arg1(a)) ⊗ copy(arg2(a))
function Base.copy!(dest::AbstractKroneckerArray, src::AbstractKroneckerArray)
    return mutate_active_args!(copy!, copy, dest, src)
end

# TODO: copyto! is typically reserved for contiguous copies (i.e. also for copying from a
# vector into an array), it might be better to not define that here.
function Base.copyto!(dest::KroneckerArray{<:Any, N}, src::KroneckerArray{<:Any, N}) where {N}
    return mutate_active_args!(copyto!, copy, dest, src)
end

function Base.convert(
        ::Type{KroneckerArray{T, N, A1, A2}}, a::AbstractKroneckerArray
    )::KroneckerArray{T, N, A1, A2} where {T, N, A1, A2}
    typeof(a) === KroneckerArray{T, N, A1, A2} && return a
    return KroneckerArray(_convert(A1, arg1(a)), _convert(A2, arg2(a)))
end

# Promote the element type if needed.
# This works around issues like:
# https://github.com/JuliaArrays/FillArrays.jl/issues/416
maybe_promot_eltype(a, elt) = eltype(a) <: elt ? a : elt.(a)

function Base.similar(
        a::AbstractKroneckerArray,
        elt::Type,
        axs::Tuple{
            CartesianProductUnitRange{<:Integer}, Vararg{CartesianProductUnitRange{<:Integer}},
        },
    )
    # TODO: Is this a good definition?
    return if isactive(arg1(a)) == isactive(arg2(a))
        similar(arg1(a), elt, arg1.(axs)) ⊗ similar(arg2(a), elt, arg2.(axs))
    elseif isactive(arg1(a))
        @assert arg2.(axs) == axes(arg2(a))
        similar(arg1(a), elt, arg1.(axs)) ⊗ maybe_promot_eltype(arg2(a), elt)
    elseif isactive(arg2(a))
        @assert arg1.(axs) == axes(arg1(a))
        maybe_promot_eltype(arg1(a), elt) ⊗ similar(arg2(a), elt, arg2.(axs))
    end
end
function Base.similar(a::AbstractKroneckerArray, elt::Type)
    # TODO: Is this a good definition?
    return if isactive(arg1(a)) == isactive(arg2(a))
        similar(arg1(a), elt) ⊗ similar(arg2(a), elt)
    elseif isactive(arg1(a))
        similar(arg1(a), elt) ⊗ maybe_promot_eltype(arg2(a), elt)
    elseif isactive(arg2(a))
        maybe_promot_eltype(arg1(a), elt) ⊗ similar(arg2(a), elt)
    end
end
function Base.similar(a::AbstractKroneckerArray)
    # TODO: Is this a good definition?
    return if isactive(arg1(a)) == isactive(arg2(a))
        similar(arg1(a)) ⊗ similar(arg2(a))
    elseif isactive(arg1(a))
        similar(arg1(a)) ⊗ arg2(a)
    elseif isactive(arg2(a))
        arg1(a) ⊗ similar(arg2(a))
    end
end

function Base.similar(
        a::AbstractArray,
        elt::Type,
        axs::Tuple{
            CartesianProductUnitRange{<:Integer}, Vararg{CartesianProductUnitRange{<:Integer}},
        },
    )
    return similar(a, elt, map(arg1, axs)) ⊗ similar(a, elt, map(arg2, axs))
end

function Base.similar(
        ::Type{ArrayT},
        axs::Tuple{
            CartesianProductUnitRange{<:Integer}, Vararg{CartesianProductUnitRange{<:Integer}},
        },
    ) where {ArrayT <: AbstractKroneckerArray}
    A1, A2 = arg1type(ArrayT), arg2type(ArrayT)
    return similar(A1, map(arg1, axs)) ⊗ similar(A2, map(arg2, axs))
end
function Base.similar(
        ::Type{ArrayT}, sz::Tuple{Int, Vararg{Int}}
    ) where {ArrayT <: AbstractKroneckerArray}
    A1, A2 = arg1type(ArrayT), arg2type(ArrayT)
    return similar(promote_type(A1, A2), sz)
end

function Base.similar(
        arrayt::Type{<:AbstractArray},
        axs::Tuple{
            CartesianProductUnitRange{<:Integer}, Vararg{CartesianProductUnitRange{<:Integer}},
        },
    )
    return similar(arrayt, map(arg1, axs)) ⊗ similar(arrayt, map(arg2, axs))
end

function Base.permutedims(a::AbstractKroneckerArray, perm)
    return permutedims(arg1(a), perm) ⊗ permutedims(arg2(a), perm)
end
using DerivableInterfaces: DerivableInterfaces, permuteddims
function DerivableInterfaces.permuteddims(a::AbstractKroneckerArray, perm)
    return permuteddims(arg1(a), perm) ⊗ permuteddims(arg2(a), perm)
end

function Base.permutedims!(dest::AbstractKroneckerArray, src::AbstractKroneckerArray, perm)
    return mutate_active_args!(
        (dest, src) -> permutedims!(dest, src, perm), Base.Fix2(permutedims, perm), dest, src
    )
end

function flatten(t::Tuple{Tuple, Tuple, Vararg{Tuple}})
    return (t[1]..., flatten(Base.tail(t))...)
end
function flatten(t::Tuple{Tuple})
    return t[1]
end
flatten(::Tuple{}) = ()
function interleave(x::Tuple, y::Tuple)
    length(x) == length(y) || throw(ArgumentError("Tuples must have the same length."))
    xy = ntuple(i -> (x[i], y[i]), length(x))
    return flatten(xy)
end
# TODO: Maybe use scalar indexing based on KroneckerProducts.jl logic for cartesian indexing:
# https://github.com/perrutquist/KroneckerProducts.jl/blob/8c0104caf1f17729eb067259ba1473986121d032/src/KroneckerProducts.jl#L59-L66
function kron_nd(a::AbstractArray{<:Any, N}, b::AbstractArray{<:Any, N}) where {N}
    a′ = reshape(a, interleave(size(a), ntuple(one, N)))
    b′ = reshape(b, interleave(ntuple(one, N), size(b)))
    c′ = permutedims(a′ .* b′, reverse(ntuple(identity, 2N)))
    sz = reverse(ntuple(i -> size(a, i) * size(b, i), N))
    return permutedims(reshape(c′, sz), reverse(ntuple(identity, N)))
end
kron_nd(a1::AbstractMatrix, a2::AbstractMatrix) = kron(a1, a2)
kron_nd(a1::AbstractVector, a2::AbstractVector) = kron(a1, a2)

# Eagerly collect arguments to make more general on GPU.
Base.collect(a::AbstractKroneckerArray) = kron_nd(collect(arg1(a)), collect(arg2(a)))
Base.collect(T::Type, a::AbstractKroneckerArray) = kron_nd(collect(T, arg1(a)), collect(T, arg2(a)))

function Base.zero(a::AbstractKroneckerArray)
    return if isactive(arg1(a)) == isactive(arg2(a))
        # TODO: Maybe this should zero both arguments?
        # This is how `a * false` would behave.
        arg1(a) ⊗ zero(arg2(a))
    elseif isactive(arg1(a))
        zero(arg1(a)) ⊗ arg2(a)
    elseif isactive(arg2(a))
        arg1(a) ⊗ zero(arg2(a))
    end
end

using DerivableInterfaces: DerivableInterfaces, zero!
function DerivableInterfaces.zero!(a::AbstractKroneckerArray)
    (isactive(arg1(a)) || isactive(arg2(a))) ||
        error("Can't mutate immutable KroneckerArray.")
    isactive(arg1(a)) && zero!(arg1(a))
    isactive(arg2(a)) && zero!(arg2(a))
    return a
end

function Base.Array{T, N}(a::AbstractKroneckerArray{S, N}) where {T, S, N}
    return convert(Array{T, N}, collect(T, a))
end

Base.size(a::AbstractKroneckerArray) = size(arg1(a)) .* size(arg2(a))

function Base.axes(a::AbstractKroneckerArray)
    return ntuple(ndims(a)) do dim
        return CartesianProductUnitRange(
            axes(arg1(a), dim) × axes(arg2(a), dim), Base.OneTo(size(a, dim))
        )
    end
end

function Base.print_array(io::IO, a::KroneckerArray)
    Base.print_array(io, arg1(a))
    println(io, "\n ⊗")
    Base.print_array(io, arg2(a))
    return nothing
end
function Base.show(io::IO, a::KroneckerArray)
    show(io, arg1(a))
    print(io, " ⊗ ")
    show(io, arg2(a))
    return nothing
end

⊗(a1::AbstractArray, a2::AbstractArray) = KroneckerArray(a1, a2)
⊗(a1::Number, a2::Number) = a1 * a2
⊗(a1::Number, a2::AbstractArray) = a1 * a2
⊗(a1::AbstractArray, a2::Number) = a1 * a2

function Base.getindex(a::KroneckerArray, i::Integer)
    return a[CartesianIndices(a)[i]]
end

using GPUArraysCore: GPUArraysCore
function Base.getindex(a::KroneckerArray{<:Any, N}, I::Vararg{Integer, N}) where {N}
    GPUArraysCore.assertscalar("getindex")
    I′ = ntuple(Val(N)) do dim
        return cartesianproduct(axes(a, dim))[I[dim]]
    end
    return a[I′...]
end

# Indexing logic.
function Base.to_indices(
        a::AbstractKroneckerArray, inds, I::Tuple{Union{CartesianPair, CartesianProduct}, Vararg}
    )
    I1 = to_indices(arg1(a), arg1.(inds), arg1.(I))
    I2 = to_indices(arg2(a), arg2.(inds), arg2.(I))
    return I1 .× I2
end

function Base.getindex(
        a::AbstractKroneckerArray{<:Any, N}, I::Vararg{Union{CartesianPair, CartesianProduct}, N}
    ) where {N}
    I′ = to_indices(a, I)
    return arg1(a)[arg1.(I′)...] ⊗ arg2(a)[arg2.(I′)...]
end
# Fix ambigiuity error.
Base.getindex(a::AbstractKroneckerArray{<:Any, 0}) = arg1(a)[] * arg2(a)[]

arg1(::Colon) = (:)
arg2(::Colon) = (:)
arg1(::Base.Slice) = (:)
arg2(::Base.Slice) = (:)
function Base.view(
        a::AbstractKroneckerArray{<:Any, N},
        I::Vararg{Union{CartesianProduct, CartesianProductUnitRange, Base.Slice, Colon}, N},
    ) where {N}
    return view(arg1(a), arg1.(I)...) ⊗ view(arg2(a), arg2.(I)...)
end
function Base.view(a::AbstractKroneckerArray{<:Any, N}, I::Vararg{CartesianPair, N}) where {N}
    return view(arg1(a), arg1.(I)...) ⊗ view(arg2(a), arg2.(I)...)
end
# Fix ambigiuity error.
Base.view(a::AbstractKroneckerArray{<:Any, 0}) = view(arg1(a)) ⊗ view(arg2(a))

function Base.:(==)(a::AbstractKroneckerArray, b::AbstractKroneckerArray)
    return arg1(a) == arg1(b) && arg2(a) == arg2(b)
end

# norm(a - b) = norm(a1 ⊗ a2 - b1 ⊗ b2)
#             = norm((a1 - b1) ⊗ a2 + b1 ⊗ (a2 - b2) + (a1 - b1) ⊗ (a2 - b2))
function dist(a::AbstractKroneckerArray, b::AbstractKroneckerArray)
    a1, a2 = arg1(a), arg2(a)
    b1, b2 = arg1(b), arg2(b)
    diff1 = a1 - b1
    diff2 = a2 - b2
    # x = (a1 - b1) ⊗ a2
    # y = b1 ⊗ (a2 - b2)
    # z = (a1 - b1) ⊗ (a2 - b2)
    xx = norm(diff1)^2 * norm(a2)^2
    yy = norm(b1)^2 * norm(diff2)^2
    zz = norm(diff1)^2 * norm(diff2)^2
    xy = real(dot(diff1, b1) * dot(a2, diff2))
    xz = real(dot(diff1, diff1) * dot(a2, diff2))
    yz = real(dot(b1, diff1) * dot(diff2, diff2))
    return sqrt(abs(xx + yy + zz + 2 * (xy + xz + yz)))
end

using LinearAlgebra: dot, promote_leaf_eltypes
function Base.isapprox(
        a::AbstractKroneckerArray, b::AbstractKroneckerArray;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(promote_leaf_eltypes(a), promote_leaf_eltypes(b), atol),
        norm::Function = norm
    )
    a1, a2 = arg1(a), arg2(a)
    b1, b2 = arg1(b), arg2(b)
    d = if a1 == b1
        norm(b1) * norm(a2 - b2)
    elseif a2 == b2
        norm(a1 - b1) * norm(b2)
    else
        # This could be defined as `KroneckerArrays.dist(a, b)`, but that might have
        # numerical precision issues so for now we just error.
        error(
            "`isapprox` not implemented for KroneckerArrays where both arguments differ. " *
                "In those cases, you can use `isapprox(collect(a), collect(b); kwargs...)`."
        )
    end
    return iszero(rtol) ? d <= atol : d <= max(atol, rtol * max(norm(a), norm(b)))
end

function Base.iszero(a::AbstractKroneckerArray)
    return iszero(arg1(a)) || iszero(arg2(a))
end
function Base.isreal(a::KroneckerArray)
    return isreal(arg1(a)) && isreal(arg2(a))
end

using DiagonalArrays: DiagonalArrays, diagonal
function DiagonalArrays.diagonal(a::KroneckerArray)
    return diagonal(arg1(a)) ⊗ diagonal(arg2(a))
end

Base.real(a::AbstractKroneckerArray{<:Real}) = a
function Base.real(a::AbstractKroneckerArray)
    if iszero(imag(arg1(a))) || iszero(imag(arg2(a)))
        return real(arg1(a)) ⊗ real(arg2(a))
    elseif iszero(real(arg1(a))) || iszero(real(arg2(a)))
        return -(imag(arg1(a)) ⊗ imag(arg2(a)))
    end
    return real(arg1(a)) ⊗ real(arg2(a)) - imag(arg1(a)) ⊗ imag(arg2(a))
end
Base.imag(a::AbstractKroneckerArray{<:Real}) = zero(a)
function Base.imag(a::AbstractKroneckerArray)
    if iszero(imag(arg1(a))) || iszero(real(arg2(a)))
        return real(arg1(a)) ⊗ imag(arg2(a))
    elseif iszero(real(arg1(a))) || iszero(imag(arg2(a)))
        return imag(arg1(a)) ⊗ real(arg2(a))
    end
    return real(arg1(a)) ⊗ imag(arg2(a)) + imag(arg1(a)) ⊗ real(arg2(a))
end

for f in [:transpose, :adjoint, :inv]
    @eval begin
        function Base.$f(a::AbstractKroneckerArray)
            return $f(arg1(a)) ⊗ $f(arg2(a))
        end
    end
end

function Base.reshape(
        a::AbstractKroneckerArray, ax::Tuple{CartesianProductUnitRange, Vararg{CartesianProductUnitRange}}
    )
    return reshape(arg1(a), map(arg1, ax)) ⊗ reshape(arg2(a), map(arg2, ax))
end

using Base.Broadcast: Broadcast, AbstractArrayStyle, BroadcastStyle, Broadcasted
struct KroneckerStyle{N, A1, A2} <: AbstractArrayStyle{N} end
arg1(::Type{<:KroneckerStyle{<:Any, A1}}) where {A1} = A1
arg1(style::KroneckerStyle) = arg1(typeof(style))
arg2(::Type{<:KroneckerStyle{<:Any, <:Any, A2}}) where {A2} = A2
arg2(style::KroneckerStyle) = arg2(typeof(style))
function KroneckerStyle{N}(a1::BroadcastStyle, a2::BroadcastStyle) where {N}
    return KroneckerStyle{N, a1, a2}()
end
function KroneckerStyle(a1::AbstractArrayStyle{N}, a2::AbstractArrayStyle{N}) where {N}
    return KroneckerStyle{N}(a1, a2)
end
function KroneckerStyle{N, A1, A2}(v::Val{M}) where {N, A1, A2, M}
    return KroneckerStyle{M, typeof(A1)(v), typeof(A2)(v)}()
end
function Base.BroadcastStyle(::Type{T}) where {T <: AbstractKroneckerArray}
    return KroneckerStyle{ndims(T)}(BroadcastStyle(arg1type(T)), BroadcastStyle(arg2type(T)))
end
function Base.BroadcastStyle(style1::KroneckerStyle{N}, style2::KroneckerStyle{N}) where {N}
    style_a = BroadcastStyle(arg1(style1), arg1(style2))
    (style_a isa Broadcast.Unknown) && return Broadcast.Unknown()
    style_b = BroadcastStyle(arg2(style1), arg2(style2))
    (style_b isa Broadcast.Unknown) && return Broadcast.Unknown()
    return KroneckerStyle{N}(style_a, style_b)
end
function Base.similar(
        bc::Broadcasted{<:KroneckerStyle{N, A1, A2}}, elt::Type, ax
    ) where {N, A1, A2}
    bc_a = Broadcasted(A1, bc.f, arg1.(bc.args), arg1.(ax))
    bc_b = Broadcasted(A2, bc.f, arg2.(bc.args), arg2.(ax))
    a = similar(bc_a, elt)
    b = similar(bc_b, elt)
    return a ⊗ b
end

function Base.map(f, a1::AbstractKroneckerArray, a_rest::AbstractKroneckerArray...)
    return Broadcast.broadcast_preserving_zero_d(f, a1, a_rest...)
end
function Base.map!(f, dest::AbstractKroneckerArray, a1::AbstractKroneckerArray, a_rest::AbstractKroneckerArray...)
    dest .= f.(a1, a_rest...)
    return dest
end

using MapBroadcast: MapBroadcast, LinearCombination, Summed
function KroneckerBroadcast(a::Summed{<:KroneckerStyle})
    f = LinearCombination(a)
    args = MapBroadcast.arguments(a)
    arg1s = arg1.(args)
    arg2s = arg2.(args)
    arg1_isunique = allequal(arg1s)
    arg2_isunique = allequal(arg2s)
    (arg1_isunique || arg2_isunique) ||
        error("This operation doesn't preserve the Kronecker structure.")
    broadcast_arg = if arg1_isunique && arg2_isunique
        isactive(first(arg1s)) ? 1 : 2
    elseif arg1_isunique
        2
    elseif arg2_isunique
        1
    end
    return if broadcast_arg == 1
        broadcasted(f, arg1s...) ⊗ first(arg2s)
    elseif broadcast_arg == 2
        first(arg1s) ⊗ broadcasted(f, arg2s...)
    end
end

function Base.copy(a::Summed{<:KroneckerStyle})
    return copy(KroneckerBroadcast(a))
end
function Base.copyto!(dest::AbstractKroneckerArray, a::Summed{<:KroneckerStyle})
    return copyto!(dest, KroneckerBroadcast(a))
end

function Broadcast.broadcasted(::KroneckerStyle, f, as...)
    return error("Arbitrary broadcasting not supported for KroneckerArray.")
end

# Linear operations.
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(+), a1, a2)
    return Summed(a1) + Summed(a2)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(-), a1, a2)
    return Summed(a1) - Summed(a2)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), c::Number, a)
    return c * Summed(a)
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), a, c::Number)
    return Summed(a) * c
end
# Fix ambiguity error.
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(*), a::Number, b::Number)
    return a * b
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(/), a, c::Number)
    return Summed(a) / c
end
function Broadcast.broadcasted(::KroneckerStyle, ::typeof(-), a)
    return -Summed(a)
end

# Rewrite rules to canonicalize broadcast expressions.
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix1{typeof(*), <:Number}, a)
    return broadcasted(style, *, f.x, a)
end
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(*), <:Number}, a)
    return broadcasted(style, *, a, f.x)
end
function Broadcast.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(/), <:Number}, a)
    return broadcasted(style, /, a, f.x)
end

# Compatibility with MapBroadcast.jl.
using MapBroadcast: MapBroadcast, MapFunction
function Base.broadcasted(
        style::KroneckerStyle, f::MapFunction{typeof(*), <:Tuple{<:Number, MapBroadcast.Arg}}, a
    )
    return broadcasted(style, *, f.args[1], a)
end
function Base.broadcasted(
        style::KroneckerStyle, f::MapFunction{typeof(*), <:Tuple{MapBroadcast.Arg, <:Number}}, a
    )
    return broadcasted(style, *, a, f.args[2])
end
function Base.broadcasted(
        style::KroneckerStyle, f::MapFunction{typeof(/), <:Tuple{MapBroadcast.Arg, <:Number}}, a
    )
    return broadcasted(style, /, a, f.args[2])
end
# Use to determine the element type of KroneckerBroadcasted.
_eltype(x) = eltype(x)
_eltype(x::Broadcasted) = Base.promote_op(x.f, _eltype.(x.args)...)

using Base.Broadcast: broadcasted
# Represents broadcast operations that can be applied Kronecker-wise,
# i.e. independently to each argument of the Kronecker product.
# Note that not all broadcast operations can be mapped to this.
struct KroneckerBroadcasted{A1, A2}
    arg1::A1
    arg2::A2
end
@inline arg1(a::KroneckerBroadcasted) = getfield(a, :arg1)
@inline arg2(a::KroneckerBroadcasted) = getfield(a, :arg2)
⊗(a1::Broadcasted, a2::Broadcasted) = KroneckerBroadcasted(a1, a2)
⊗(a1::Broadcasted, a2) = KroneckerBroadcasted(a1, a2)
⊗(a1, a2::Broadcasted) = KroneckerBroadcasted(a1, a2)
Broadcast.materialize(a::KroneckerBroadcasted) = copy(a)
Broadcast.materialize!(dest, a::KroneckerBroadcasted) = copyto!(dest, a)
Broadcast.broadcastable(a::KroneckerBroadcasted) = a
Base.copy(a::KroneckerBroadcasted) = copy(arg1(a)) ⊗ copy(arg2(a))
function Base.copyto!(dest::KroneckerArray, src::KroneckerBroadcasted)
    return mutate_active_args!(copyto!, copy, dest, src)
end
function Base.eltype(a::KroneckerBroadcasted)
    a1 = arg1(a)
    a2 = arg2(a)
    return Base.promote_op(*, _eltype(a1), _eltype(a2))
end
function Base.axes(a::KroneckerBroadcasted)
    ax1 = axes(arg1(a))
    ax2 = axes(arg2(a))
    return cartesianrange.(ax1 .× ax2)
end

function Base.BroadcastStyle(
        ::Type{<:KroneckerBroadcasted{A1, A2}}
    ) where {StyleA1, StyleA2, A1 <: Broadcasted{StyleA1}, A2 <: Broadcasted{StyleA2}}
    @assert ndims(A1) == ndims(A2)
    N = ndims(A1)
    return KroneckerStyle{N}(StyleA1(), StyleA2())
end

# Operations that preserve the Kronecker structure.
for f in [:identity, :conj]
    @eval begin
        function Broadcast.broadcasted(
                ::KroneckerStyle{<:Any, A1, A2}, ::typeof($f), a
            ) where {A1, A2}
            return broadcasted(A1, $f, arg1(a)) ⊗ broadcasted(A2, $f, arg2(a))
        end
    end
end
