"""
    abstract type AbstractKroneckerArray{T, N} <: AbstractArray{T, N} end

Abstract supertype for arrays that have a kronecker product structure,
i.e. that can be written as `AB = A ⊗ B`.
"""
abstract type AbstractKroneckerArray{T, N} <: AbstractArray{T, N} end

const AbstractKroneckerVector{T} = AbstractKroneckerArray{T, 1}
const AbstractKroneckerMatrix{T} = AbstractKroneckerArray{T, 2}

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
_construct(A::Type{<:Diagonal}, a::AbstractMatrix) = A(diag(a))
function _convert(A::Type{<:Diagonal}, a::AbstractMatrix)
    LinearAlgebra.checksquare(a)
    return isdiag(a) ? _construct(A, a) : throw(InexactError(:convert, A, a))
end

struct KroneckerArray{T, N, A <: AbstractArray{T, N}, B <: AbstractArray{T, N}} <:
    AbstractKroneckerArray{T, N}
    a::A
    b::B
end
function KroneckerArray(a::AbstractArray, b::AbstractArray)
    ndims(a) == ndims(b) ||
        throw(
        DimensionMismatch(
            "Kronecker product requires arrays of the same number of dimensions."
        )
    )
    elt = promote_type(eltype(a), eltype(b))
    return _convert(AbstractArray{elt}, a) ⊗ _convert(AbstractArray{elt}, b)
end

const KroneckerMatrix{T, A <: AbstractMatrix{T}, B <: AbstractMatrix{T}} =
    KroneckerArray{T, 2, A, B}
const KroneckerVector{T, A <: AbstractVector{T}, B <: AbstractVector{T}} =
    KroneckerArray{T, 1, A, B}

kroneckerfactors(ab::KroneckerArray) = (ab.a, ab.b)
kroneckerfactortypes(::Type{KroneckerArray{T, N, A, B}}) where {T, N, A, B} = (A, B)

function mutate_active_args!(f!, f, C, A)
    Ca, Cb = kroneckerfactors(C)
    Aa, Ab = kroneckerfactors(A)
    (isactive(Ca) || isactive(Cb)) ||
        error("Can't mutate immutable KroneckerArray.")
    if isactive(Ca)
        f!(Ca, Aa)
    else
        Ca == f(Aa) || error("Immutable arguments aren't equal.")
    end
    if isactive(Cb)
        f!(Cb, Ab)
    else
        Cb == f(Ab) || error("Immutable arguments aren't equal.")
    end
    return C
end

function Adapt.adapt_structure(to, ab::AbstractKroneckerArray)
    # TODO: Is this a good definition? It is similar to the definition of `similar`.
    a, b = kroneckerfactors(ab)
    return if isactive(a) == isactive(b)
        Adapt.adapt(to, a) ⊗ Adapt.adapt(to, b)
    elseif isactive(a)
        Adapt.adapt(to, a) ⊗ b
    elseif isactive(b)
        a ⊗ Adapt.adapt(to, b)
    end
end

Base.copy(a::AbstractKroneckerArray) = ⊗(copy.(kroneckerfactors(a))...)
function Base.copy!(dest::AbstractKroneckerArray, src::AbstractKroneckerArray)
    return mutate_active_args!(copy!, copy, dest, src)
end

# TODO: copyto! is typically reserved for contiguous copies (i.e. also for copying from a
# vector into an array), it might be better to not define that here.
function Base.copyto!(
        dest::AbstractKroneckerArray{<:Any, N},
        src::AbstractKroneckerArray{<:Any, N}
    ) where {N}
    return mutate_active_args!(copyto!, copy, dest, src)
end

function Base.convert(
        ::Type{KroneckerArray{T, N, A, B}}, ab::AbstractKroneckerArray
    )::KroneckerArray{T, N, A, B} where {T, N, A, B}
    typeof(ab) === KroneckerArray{T, N, A, B} && return ab
    a, b = kroneckerfactors(ab)
    return KroneckerArray(_convert(A, a), _convert(B, b))
end

# Promote the element type if needed.
# This works around issues like:
# https://github.com/JuliaArrays/FillArrays.jl/issues/416
maybe_promot_eltype(a, elt) = eltype(a) <: elt ? a : elt.(a)

function Base.similar(
        ab::AbstractKroneckerArray,
        elt::Type,
        axs::Tuple{
            CartesianProductUnitRange{<:Integer},
            Vararg{CartesianProductUnitRange{<:Integer}},
        }
    )
    # TODO: Is this a good definition?
    a, b = kroneckerfactors(ab)
    return if isactive(a) == isactive(b)
        similar(a, elt, kroneckerfactors.(axs, 1)) ⊗
            similar(b, elt, kroneckerfactors.(axs, 2))
    elseif isactive(a)
        @assert kroneckerfactors.(axs, 2) == axes(b)
        similar(a, elt, kroneckerfactors.(axs, 1)) ⊗ maybe_promot_eltype(b, elt)
    elseif isactive(b)
        @assert kroneckerfactors.(axs, 1) == axes(a)
        maybe_promot_eltype(a, elt) ⊗ similar(b, elt, kroneckerfactors.(axs, 2))
    end
end
function Base.similar(ab::AbstractKroneckerArray, elt::Type)
    # TODO: Is this a good definition?
    a, b = kroneckerfactors(ab)
    return if isactive(a) == isactive(b)
        similar(a, elt) ⊗ similar(b, elt)
    elseif isactive(a)
        similar(a, elt) ⊗ maybe_promot_eltype(b, elt)
    elseif isactive(b)
        maybe_promot_eltype(a, elt) ⊗ similar(b, elt)
    end
end
function Base.similar(ab::AbstractKroneckerArray)
    # TODO: Is this a good definition?
    a, b = kroneckerfactors(ab)
    return if isactive(a) == isactive(b)
        similar(a) ⊗ similar(b)
    elseif isactive(a)
        similar(a) ⊗ b
    elseif isactive(b)
        a ⊗ similar(b)
    end
end

function Base.similar(
        a::AbstractArray,
        elt::Type,
        axs::Tuple{
            CartesianProductUnitRange{<:Integer},
            Vararg{CartesianProductUnitRange{<:Integer}},
        }
    )
    return similar(a, elt, kroneckerfactors.(axs, 1)) ⊗
        similar(a, elt, kroneckerfactors.(axs, 2))
end

function Base.similar(
        ::Type{ArrayT},
        axs::Tuple{
            CartesianProductUnitRange{<:Integer},
            Vararg{CartesianProductUnitRange{<:Integer}},
        }
    ) where {ArrayT <: AbstractKroneckerArray}
    A, B = kroneckerfactortypes(ArrayT)
    return similar(A, kroneckerfactors.(axs, 1)) ⊗ similar(B, kroneckerfactors.(axs, 2))
end
function Base.similar(
        ::Type{ArrayT},
        sz::Tuple{Int, Vararg{Int}}
    ) where {ArrayT <: AbstractKroneckerArray}
    A, B = kroneckerfactortypes(ArrayT)
    return similar(promote_type(A, B), sz)
end

function Base.similar(
        arrayt::Type{<:AbstractArray},
        axs::Tuple{
            CartesianProductUnitRange{<:Integer},
            Vararg{CartesianProductUnitRange{<:Integer}},
        }
    )
    return similar(arrayt, kroneckerfactors.(axs, 1)) ⊗
        similar(arrayt, kroneckerfactors.(axs, 2))
end

function Base.permutedims(ab::AbstractKroneckerArray, perm)
    return ⊗(permutedims.(kroneckerfactors(ab), (perm,))...)
end
function FunctionImplementations.permuteddims(ab::AbstractKroneckerArray, perm)
    return ⊗(FunctionImplementations.permuteddims.(kroneckerfactors(ab), (perm,))...)
end

function Base.permutedims!(dest::AbstractKroneckerArray, src::AbstractKroneckerArray, perm)
    return mutate_active_args!(
        (dest, src) -> permutedims!(dest, src, perm), Base.Fix2(permutedims, perm), dest,
        src
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
Base.collect(ab::AbstractKroneckerArray) = kron_nd(collect.(kroneckerfactors(ab))...)
function Base.collect(T::Type, ab::AbstractKroneckerArray)
    return kron_nd(collect.(T, kroneckerfactors(ab))...)
end

function Base.zero(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    return if isactive(a) == isactive(b)
        # TODO: Maybe this should zero both arguments?
        # This is how `a * false` would behave.
        a ⊗ zero(b)
    elseif isactive(a)
        zero(a) ⊗ b
    elseif isactive(b)
        a ⊗ zero(b)
    end
end

function FunctionImplementations.zero!(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    (isactive(a) || isactive(b)) || error("Can't mutate immutable KroneckerArray.")
    isactive(a) && FunctionImplementations.zero!(a)
    isactive(b) && FunctionImplementations.zero!(b)
    return ab
end

Base.Array{T, N}(a::AbstractKroneckerArray) where {T, N} = convert(Array{T, N}, collect(a))

Base.size(ab::AbstractKroneckerArray) = broadcast(*, size.(kroneckerfactors(ab))...)

function Base.axes(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    return axes(a) .× axes(b)
end

function Base.print_array(io::IO, ab::KroneckerArray)
    a, b = kroneckerfactors(ab)
    Base.print_array(io, a)
    println(io, "\n ⊗")
    Base.print_array(io, b)
    return nothing
end
function Base.show(io::IO, ab::KroneckerArray)
    a, b = kroneckerfactors(ab)
    show(io, a)
    print(io, " ⊗ ")
    show(io, b)
    return nothing
end

⊗(a1::AbstractArray, a2::AbstractArray) = KroneckerArray(a1, a2)
⊗(a1::Number, a2::Number) = a1 * a2
⊗(a1::Number, a2::AbstractArray) = a1 * a2
⊗(a1::AbstractArray, a2::Number) = a1 * a2

function Base.getindex(a::AbstractKroneckerArray{<:Any, N}, I::Vararg{Int, N}) where {N}
    GPUArraysCore.assertscalar("getindex")
    I′ = ntuple(Val(N)) do dim
        return cartesianproduct(kroneckerfactors(axes(a, dim))...)[I[dim]]
    end
    return a[I′...]
end

# Indexing logic.
function Base.to_indices(
        ab::AbstractKroneckerArray, inds,
        I::Tuple{Union{CartesianPair, CartesianProduct}, Vararg}
    )
    a, b = kroneckerfactors(ab)
    I1 = to_indices(a, kroneckerfactors.(inds, 1), kroneckerfactors.(I, 1))
    I2 = to_indices(b, kroneckerfactors.(inds, 2), kroneckerfactors.(I, 2))
    return I1 .× I2
end

function Base.getindex(
        ab::AbstractKroneckerArray{<:Any, N},
        I::Vararg{Union{CartesianPair, CartesianProduct, CartesianProductUnitRange}, N}
    ) where {N}
    I′ = to_indices(ab, I)
    a, b = kroneckerfactors(ab)
    return a[kroneckerfactors.(I′, 1)...] ⊗ b[kroneckerfactors.(I′, 2)...]
end

# Fix ambigiuity error.
Base.getindex(ab::AbstractKroneckerArray{<:Any, 0}) = *(getindex.(kroneckerfactors(ab))...)

kroneckerfactors(::Colon) = ((:), (:))
kroneckerfactors(::Base.Slice) = ((:), (:))

function Base.view(
        ab::AbstractKroneckerArray{<:Any, N},
        I::Vararg{Union{CartesianProduct, CartesianProductUnitRange, Base.Slice, Colon}, N}
    ) where {N}
    a, b = kroneckerfactors(ab)
    Ia = kroneckerfactors.(I, 1)
    Ib = kroneckerfactors.(I, 2)
    return view(a, Ia...) ⊗ view(b, Ib...)
end
function Base.view(
        ab::AbstractKroneckerArray{<:Any, N},
        I::Vararg{CartesianPair, N}
    ) where {N}
    a, b = kroneckerfactors(ab)
    Ia = kroneckerfactors.(I, 1)
    Ib = kroneckerfactors.(I, 2)
    return view(a, Ia...) ⊗ view(b, Ib...)
end
# Fix ambigiuity error.
Base.view(ab::AbstractKroneckerArray{<:Any, 0}) = ⊗(view.(kroneckerfactors(ab))...)

function Base.:(==)(ab::AbstractKroneckerArray, cd::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    c, d = kroneckerfactors(cd)
    return a == c && b == d
end

# norm(a - b) = norm(a1 ⊗ a2 - b1 ⊗ b2)
#             = norm((a1 - b1) ⊗ a2 + b1 ⊗ (a2 - b2) + (a1 - b1) ⊗ (a2 - b2))
function dist_kronecker(a::AbstractKroneckerArray, b::AbstractKroneckerArray)
    a1, a2 = kroneckerfactors(a)
    b1, b2 = kroneckerfactors(b)
    diff1 = a1 - b1
    diff2 = a2 - b2
    # x = (a1 - b1) ⊗ a2
    # y = b1 ⊗ (a2 - b2)
    # z = (a1 - b1) ⊗ (a2 - b2)
    xx = LinearAlgebra.norm(diff1)^2 * LinearAlgebra.norm(a2)^2
    yy = LinearAlgebra.norm(b1)^2 * LinearAlgebra.norm(diff2)^2
    zz = LinearAlgebra.norm(diff1)^2 * LinearAlgebra.norm(diff2)^2
    xy = real(LinearAlgebra.dot(diff1, b1) * LinearAlgebra.dot(a2, diff2))
    xz = real(LinearAlgebra.dot(diff1, diff1) * LinearAlgebra.dot(a2, diff2))
    yz = real(LinearAlgebra.dot(b1, diff1) * LinearAlgebra.dot(diff2, diff2))
    # `abs` is used in case there are negative values due to floating point roundoff errors.
    return sqrt(abs(xx + yy + zz + 2 * (xy + xz + yz)))
end

using LinearAlgebra: dot, promote_leaf_eltypes
function Base.isapprox(
        a::AbstractKroneckerArray, b::AbstractKroneckerArray; atol::Real = 0,
        rtol::Real = Base.rtoldefault(promote_leaf_eltypes(a), promote_leaf_eltypes(b), atol)
    )
    a1, a2 = kroneckerfactors(a)
    b1, b2 = kroneckerfactors(b)
    if a1 == b1
        return isapprox(a2, b2; atol = atol / LinearAlgebra.norm(a1), rtol)
    elseif a2 == b2
        return isapprox(a1, b1; atol = atol / LinearAlgebra.norm(a2), rtol)
    else
        # This could be defined as:
        # ```julia
        # d = KroneckerArrays.dist_kronecker(a, b)
        # iszero(rtol) ? d <= atol : d <= max(atol, rtol * max(norm(a), norm(b)))
        # ```
        # but that might have numerical precision issues so for now we just error.
        throw(
            ArgumentError(
                "`isapprox` not implemented for KroneckerArrays where both arguments differ. " *
                    "In those cases, you can use `isapprox(collect(a), collect(b); kwargs...)`."
            )
        )
    end
end

function Base.iszero(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    return iszero(a) || iszero(b)
end
function Base.isreal(ab::KroneckerArray)
    a, b = kroneckerfactors(ab)
    return isreal(a) && isreal(b)
end

function DiagonalArrays.diagonal(ab::KroneckerArray)
    return ⊗(DiagonalArrays.diagonal.(kroneckerfactors(ab))...)
end

Base.real(ab::AbstractKroneckerArray{<:Real}) = ab
# TODO: the extra checks here are probably as expensive as the general case
function Base.real(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    if iszero(imag(a)) || iszero(imag(b))
        return real(a) ⊗ real(b)
    elseif iszero(real(a)) || iszero(real(b))
        return -(imag(a) ⊗ imag(b))
    end
    return real(a) ⊗ real(b) - imag(a) ⊗ imag(b)
end

Base.imag(ab::AbstractKroneckerArray{<:Real}) = zero(ab)
# TODO: the extra checks here are probably as expensive as the general case
function Base.imag(ab::AbstractKroneckerArray)
    a, b = kroneckerfactors(ab)
    if iszero(imag(a)) || iszero(real(b))
        return real(a) ⊗ imag(b)
    elseif iszero(real(a)) || iszero(imag(b))
        return imag(a) ⊗ real(b)
    end
    return real(a) ⊗ imag(b) + imag(a) ⊗ real(b)
end

for f in (:transpose, :adjoint, :inv)
    @eval Base.$f(ab::AbstractKroneckerArray) = ⊗($f.(kroneckerfactors(ab))...)
end

function Base.reshape(
        ab::AbstractKroneckerArray,
        ax::Tuple{CartesianProductUnitRange, Vararg{CartesianProductUnitRange}}
    )
    a, b = kroneckerfactors(ab)
    return reshape(a, kroneckerfactors.(ax, 1)) ⊗ reshape(b, kroneckerfactors.(ax, 2))
end

function Base.fill!(ab::AbstractKroneckerArray, v)
    a, b = kroneckerfactors(ab)
    fill!(a, √v)
    fill!(b, √v)
    return ab
end
function Base.fill!(ab::AbstractKroneckerMatrix, v)
    a, b = kroneckerfactors(ab)
    (!isactive(a) && isone(a)) && (fill!(b, v); return ab)
    (!isactive(b) && isone(b)) && (fill!(a, v); return ab)
    fill!(a, √v)
    fill!(b, √v)
    return ab
end
function Base.fill!(ab::AbstractKroneckerVector, v)
    a, b = kroneckerfactors(ab)
    (!isactive(a) && all(isone, a)) && (fill!(b, v); return ab)
    (!isactive(b) && all(isone, b)) && (fill!(a, v); return ab)
    fill!(a, √v)
    fill!(b, √v)
    return ab
end

using Base.Broadcast: AbstractArrayStyle, Broadcast, BroadcastStyle, Broadcasted

struct KroneckerStyle{N, A, B} <: BC.AbstractArrayStyle{N} end

kroneckerfactors(::Type{KroneckerStyle{N, A, B}}) where {N, A, B} = (A, B)
kroneckerfactors(style::KroneckerStyle) = kroneckerfactors(typeof(style))

function KroneckerStyle{N}(A::BroadcastStyle, B::BroadcastStyle) where {N}
    return KroneckerStyle{N, A, B}()
end
function KroneckerStyle(A::AbstractArrayStyle{N}, B::AbstractArrayStyle{N}) where {N}
    return KroneckerStyle{N}(A, B)
end
function KroneckerStyle{N, A, B}(v::Val{M}) where {N, A, B, M}
    return KroneckerStyle{M, typeof(A)(v), typeof(B)(v)}()
end

function Base.BroadcastStyle(::Type{T}) where {T <: AbstractKroneckerArray}
    return KroneckerStyle{ndims(T)}(BroadcastStyle.(kroneckerfactortypes(T))...)
end
function Base.BroadcastStyle(style1::KroneckerStyle{N}, style2::KroneckerStyle{N}) where {N}
    A1, B1 = kroneckerfactors(style1)
    A2, B2 = kroneckerfactors(style2)
    style_a = BroadcastStyle(A1, A2)
    (style_a isa BC.Unknown) && return BC.Unknown()
    style_b = BroadcastStyle(B1, B2)
    (style_b isa BC.Unknown) && return BC.Unknown()
    return KroneckerStyle{N}(style_a, style_b)
end

function Base.similar(
        bc::BC.Broadcasted{<:KroneckerStyle{N, A, B}}, elt::Type, ax
    ) where {N, A, B}
    bc_a = BC.Broadcasted(A, bc.f, kroneckerfactors.(bc.args, 1), kroneckerfactors.(ax, 1))
    a = similar(bc_a, elt)
    bc_b = BC.Broadcasted(B, bc.f, kroneckerfactors.(bc.args, 1), kroneckerfactors.(ax, 2))
    b = similar(bc_b, elt)
    return a ⊗ b
end

function Base.map(f, a1::AbstractKroneckerArray, a_rest::AbstractKroneckerArray...)
    return Broadcast.broadcast_preserving_zero_d(f, a1, a_rest...)
end
function Base.map!(
        f,
        dest::AbstractKroneckerArray,
        a1::AbstractKroneckerArray,
        a_rest::AbstractKroneckerArray...
    )
    dest .= f.(a1, a_rest...)
    return dest
end

function KroneckerBroadcast(a::Summed{<:KroneckerStyle})
    f = LinearCombination(a)
    args = MapBroadcast.arguments(a)
    arg1s = kroneckerfactors.(args, 1)
    arg2s = kroneckerfactors.(args, 2)
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
        BC.broadcasted(f, arg1s...) ⊗ first(arg2s)
    elseif broadcast_arg == 2
        first(arg1s) ⊗ BC.broadcasted(f, arg2s...)
    end
end

Base.copy(a::Summed{<:KroneckerStyle}) = copy(KroneckerBroadcast(a))
function Base.copyto!(dest::AbstractKroneckerArray, a::Summed{<:KroneckerStyle})
    return copyto!(dest, KroneckerBroadcast(a))
end

function BC.broadcasted(::KroneckerStyle, f, as...)
    return error("Arbitrary broadcasting not supported for KroneckerStyle.")
end

# Linear operations.
BC.broadcasted(::KroneckerStyle, ::typeof(+), a1, a2) = Summed(a1) + Summed(a2)
BC.broadcasted(::KroneckerStyle, ::typeof(-), a1, a2) = Summed(a1) - Summed(a2)
BC.broadcasted(::KroneckerStyle, ::typeof(*), c::Number, a) = c * Summed(a)
BC.broadcasted(::KroneckerStyle, ::typeof(*), a, c::Number) = Summed(a) * c

# Fix ambiguity error.
BC.broadcasted(::KroneckerStyle, ::typeof(*), a::Number, b::Number) = a * b
BC.broadcasted(::KroneckerStyle, ::typeof(/), a, c::Number) = Summed(a) / c
BC.broadcasted(::KroneckerStyle, ::typeof(-), a) = -Summed(a)

# Rewrite rules to canonicalize broadcast expressions.
function BC.broadcasted(style::KroneckerStyle, f::Base.Fix1{typeof(*), <:Number}, a)
    return BC.broadcasted(style, *, f.x, a)
end
function BC.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(*), <:Number}, a)
    return BC.broadcasted(style, *, a, f.x)
end
function BC.broadcasted(style::KroneckerStyle, f::Base.Fix2{typeof(/), <:Number}, a)
    return BC.broadcasted(style, /, a, f.x)
end

# Compatibility with MapBroadcast.jl.
function BC.broadcasted(
        style::KroneckerStyle,
        f::MapFunction{typeof(*), <:Tuple{<:Number, MapBroadcast.Arg}},
        a
    )
    return BC.broadcasted(style, *, f.args[1], a)
end
function BC.broadcasted(
        style::KroneckerStyle,
        f::MapFunction{typeof(*), <:Tuple{MapBroadcast.Arg, <:Number}},
        a
    )
    return BC.broadcasted(style, *, a, f.args[2])
end
function BC.broadcasted(
        style::KroneckerStyle,
        f::MapFunction{typeof(/), <:Tuple{MapBroadcast.Arg, <:Number}},
        a
    )
    return BC.broadcasted(style, /, a, f.args[2])
end

# Use to determine the element type of KroneckerBroadcasted.
_eltype(x) = eltype(x)
_eltype(x::BC.Broadcasted) = Base.promote_op(x.f, _eltype.(x.args)...)

# Represents broadcast operations that can be applied Kronecker-wise,
# i.e. independently to each argument of the Kronecker product.
# Note that not all broadcast operations can be mapped to this.
struct KroneckerBroadcasted{A, B}
    a::A
    b::B
end

kroneckerfactors(ab::KroneckerBroadcasted) = ab.a, ab.b
kroneckerfactortypes(::Type{KroneckerBroadcasted{A, B}}) where {A, B} = (A, B)

⊗(a1::BC.Broadcasted, a2::BC.Broadcasted) = KroneckerBroadcasted(a1, a2)
⊗(a1::BC.Broadcasted, a2) = KroneckerBroadcasted(a1, a2)
⊗(a1, a2::BC.Broadcasted) = KroneckerBroadcasted(a1, a2)

BC.materialize(a::KroneckerBroadcasted) = copy(a)
BC.materialize!(dest, a::KroneckerBroadcasted) = copyto!(dest, a)
BC.broadcastable(a::KroneckerBroadcasted) = a

Base.copy(ab::KroneckerBroadcasted) = ⊗(copy.(kroneckerfactors(ab))...)
function Base.copyto!(dest::AbstractKroneckerArray, src::KroneckerBroadcasted)
    return mutate_active_args!(copyto!, copy, dest, src)
end
function Base.eltype(ab::KroneckerBroadcasted)
    a, b = kroneckerfactors(ab)
    return Base.promote_op(*, _eltype(a), _eltype(b))
end
function Base.axes(ab::KroneckerBroadcasted)
    ax1, ax2 = axes.(kroneckerfactors(ab))
    return cartesianrange.(ax1, ax2)
end

function Base.BroadcastStyle(
        ::Type{<:KroneckerBroadcasted{A, B}}
    ) where {StyleA1, StyleA2, A <: BC.Broadcasted{StyleA1}, B <: BC.Broadcasted{StyleA2}}
    @assert ndims(A) == ndims(B)
    N = ndims(A)
    return KroneckerStyle{N}(StyleA1(), StyleA2())
end

# Operations that preserve the Kronecker structure.
for f in (:identity, :conj)
    @eval function BC.broadcasted(
            ::KroneckerStyle{<:Any, A, B},
            ::typeof($f),
            ab
        ) where {A, B}
        a, b = kroneckerfactors(ab)
        return BC.broadcasted(A, $f, a) ⊗ BC.broadcasted(B, $f, b)
    end
end
