# Cartesian product types
# -----------------------
# This file contains several different definitions for cartesian product objects.
# The multiple types are required to get around Julia's type system not allowing parametric
# supertypes.

"""
    CartesianPair(a, b)

Represents a single element, the cartesian product of two arbitrary objects `a` and `b`.
"""
struct CartesianPair{A, B}
    a::A
    b::B
end

"""
    CartesianProduct(a::AbstractVector, b::AbstractVector)

Represents the cartesian product of two collections `a` and `b`.
"""
struct CartesianProduct{TA, TB, A <: AbstractVector{TA}, B <: AbstractVector{TB}} <:
    AbstractVector{CartesianPair{TA, TB}}
    a::A
    b::B
end

"""
    CartesianProductVector(a::AbstractVector, b::AbstractVector, values::AbstractVector{T}) <: AbstractVector{T}

Similar to the [`CartesianProduct`](@ref), this represents the cartesian product of two collections `a` and `b`.
However, as a vector it will behave as `values`, rather than `CartesianPair`s of the elements of `a` and `b`.
"""
struct CartesianProductVector{T, A, B, V <: AbstractVector{T}} <: AbstractVector{T}
    a::A
    b::B
    values::V

    function CartesianProductVector{T, A, B, V}(
            a::A, b::B, values::V
        ) where {T, A, B, V <: AbstractVector{T}}
        length(a) * length(b) == length(values) || throw(DimensionMismatch())
        return new{T, A, B, V}(a, b, values)
    end
end
CartesianProductVector(a, b, values::AbstractVector{T}) where {T} =
    CartesianProductVector{T, typeof(a), typeof(b), typeof(values)}(a, b, values)

"""
    CartesianProductUnitRange(a::AbstractUnitRange, b::AbstractUnitRange, range::AbstractUnitRange{T}) <: AbstractUnitRange{T}

Similar to [`CartesianProductVector`](@ref), this represents the cartesian product of two ranges `a` and `b`.
However, as a range it will behave as `range`, rather than `CartesianPair`s of the elements of `a` and `b`.
"""
struct CartesianProductUnitRange{
        T, A <: AbstractUnitRange{T}, B <: AbstractUnitRange{T}, R <: AbstractUnitRange{T},
    } <: AbstractUnitRange{T}
    a::A
    b::B
    range::R

    function CartesianProductUnitRange{T, A, B, R}(
            a::A, b::B, range::R
        ) where {T, A <: AbstractUnitRange{T}, B <: AbstractUnitRange{T}, R <: AbstractUnitRange{T}}
        length(a) * length(b) == length(range) || throw(DimensionMismatch())
        return new{T, A, B, R}(a, b, range)
    end
end
CartesianProductUnitRange(a::AbstractUnitRange{T}, b::AbstractUnitRange{T}, range::AbstractUnitRange{T}) where {T} =
    CartesianProductUnitRange{T, typeof(a), typeof(b), typeof(range)}(a, b, range)
CartesianProductUnitRange(a::AbstractUnitRange{T}, b::AbstractUnitRange{T}) where {T} =
    CartesianProductUnitRange(a, b, Base.OneTo{T}(length(a) * length(b)))

const CartesianProductOneTo{T, A <: AbstractUnitRange{T}, B <: AbstractUnitRange{T}} =
    CartesianProductUnitRange{T, A, B, Base.OneTo{T}}

const AnyCartesian = Union{CartesianPair, CartesianProduct, CartesianProductVector, CartesianProductUnitRange}

# Utility constructors
# --------------------
@doc """
    ×(args...)
    times(args...)

Construct an object that represents the Cartesian product of the provided `args`.
By default this constructs the singular [`CartesianPair`](@ref) for unknown values, while attempting to promote to more structured types wherever possible.
See also [`CartesianProduct`](@ref), [`CartesianProductVector`](@ref) and [`CartesianProductUnitRange`](@ref).
""" times
# implement multi-argument version through a left fold
times(x) = x
times(x, y, z...) = foldl(times, (x, y, z...))
const × = times # unicode alternative
# fallback definition for cartesian product
×(a, b) = CartesianPair(a, b)

# attempt to construct most specific type
×(a::AbstractVector, b::AbstractVector) = cartesianproduct(a, b)
×(a::AbstractUnitRange{T}, b::AbstractUnitRange{T}) where {T} = cartesianrange(a, b)

@doc """
    cartesianproduct(a::AbstractVector, b::AbstractVector, [values::AbstractVector])::AbstractVector

Construct an `AbstractVector` that represents the cartesian product `a × b`, but behaves as `values`.
This behaves similar to [`×`](@ref), but forces promotion to a `AbstractVector`.
""" cartesianproduct

cartesianproduct(a::AbstractVector, b::AbstractVector) = CartesianProduct(a, b)
cartesianproduct(a::AbstractVector, b::AbstractVector, values::AbstractVector) = CartesianProductVector(a, b, values)
cartesianproduct(p::CartesianPair) = cartesianproduct(kroneckerfactors(p)...)
cartesianproduct(p::CartesianPair, values::AbstractVector) = cartesianproduct(kroneckerfactors(p)..., values)

@doc """
    cartesianrange(a::AbstractUnitRange, b::AbstractUnitRange, [range::AbstractUnitRange])::AbstractUnitRange

Construct a `UnitRange` that represents the cartesian product `a × b`, but behaves as `range`.
This behaves similar to [`×`](@ref), but forces promotion to a `AbstractUnitRange`.
""" cartesianrange

to_product_indices(a::AbstractVector) = a
to_product_indices(i::Integer) = Base.OneTo(i)

cartesianrange(a, b) = CartesianProductUnitRange(to_product_indices(a), to_product_indices(b))
cartesianrange(a, b, range::AbstractUnitRange) = CartesianProductUnitRange(to_product_indices(a), to_product_indices(b), range)
cartesianrange(p::Union{CartesianPair, CartesianProduct}) = cartesianrange(kroneckerfactors(p)...)
cartesianrange(p::Union{CartesianPair, CartesianProduct}, range::AbstractUnitRange) = cartesianrange(kroneckerfactors(p)..., range)

# KroneckerArrays interface
# -------------------------
kroneckerfactors(ab::AnyCartesian) = (ab.a, ab.b)
kroneckerfactortypes(::Type{T}) where {T <: CartesianPair} = fieldtypes(T)
kroneckerfactortypes(::Type{T}) where {T <: CartesianProduct} = kroneckerfactortypes(eltype(T))
kroneckerfactortypes(::Type{<:CartesianProductVector{T, A, B}}) where {T, A, B} = (A, B)
kroneckerfactortypes(::Type{<:CartesianProductUnitRange{T, A, B}}) where {T, A, B} = (A, B)

@doc """
    unproduct(a)

For an object that holds a cartesian product of indices and their corresponding values,
this function removes the cartesian product layer and returns only the values.
""" unproduct

unproduct(ab::CartesianProduct) = collect(ab)
unproduct(ab::CartesianProductVector) = ab.values
unproduct(ab::CartesianProductUnitRange) = ab.range

# AbstractVector interface
# ------------------------
Base.size(a::CartesianProduct) = (prod(length, kroneckerfactors(a)),)
Base.size(a::CartesianProductVector) = size(unproduct(a))
Base.size(a::CartesianProductUnitRange) = size(unproduct(a))

# function Base.axes(r::CartesianProduct)
#     prod_ax = only.(axes.(kroneckerfactors(r)))
#     return (cartesianrange(prod_ax...),)
# end
function Base.axes(r::CartesianProductVector)
    prod_ax = only.(axes.(kroneckerfactors(r)))
    return (cartesianrange(prod_ax..., only(axes(r.values))),)
end
function Base.axes(r::CartesianProductUnitRange)
    prod_ax = only.(axes.(kroneckerfactors(r)))
    return (cartesianrange(prod_ax..., only(axes(r.range))),)
end
# TODO: add comment why this is here
Base.axes(S::Base.Slice{<:CartesianProductOneTo}) = (S.indices,)
Base.axes1(S::Base.Slice{<:CartesianProductOneTo}) = S.indices
Base.unsafe_indices(S::Base.Slice{<:CartesianProductOneTo}) = (S.indices,)

Base.copy(a::CartesianProduct) = ×(copy.(kroneckerfactors(a)...)...)
Base.copy(a::CartesianProductVector) = cartesianproduct(copy.(kroneckerfactors(a))..., copy(unproduct(a)))

@inline Base.getindex(a::CartesianProduct, i::CartesianProduct) =
    ×(Base.getindex.(kroneckerfactors(a), kroneckerfactors(i))...)
@inline Base.getindex(a::CartesianProduct, i::CartesianPair) =
    ×(Base.getindex.(kroneckerfactors(a), kroneckerfactors(i))...)

Base.@propagate_inbounds function Base.getindex(a::CartesianProduct, i::Int)
    I = Tuple(CartesianIndices(reverse(length.(kroneckerfactors(a))))[i])
    return a[I[2] × I[1]]
end
@inline Base.getindex(r::CartesianProductVector, i::Int) = r.values[i]

Base.@propagate_inbounds function Base.getindex(a::CartesianProductUnitRange, i::CartesianProductUnitRange)
    return cartesianrange(Base.getindex.(kroneckerfactors(a), kroneckerfactors(i))..., a.range[i.range])
end

function Base.getindex(a::CartesianProductUnitRange, I::CartesianProduct)
    return cartesianproduct(Base.getindex.(kroneckerfactors(a), kroneckerfactors(I))..., map(Base.Fix1(getindex, a), I))
end

# Reverse map from CartesianPair to linear index in the range.
Base.@propagate_inbounds function Base.getindex(inds::CartesianProductUnitRange, i::CartesianPair)
    indsa, indsb = kroneckerfactors(inds)
    ia, ib = kroneckerfactors(i)
    i′ = CartesianIndex(findfirst(==(ib), indsb), findfirst(==(ia), indsa))
    i_linear = LinearIndices(reverse(length.(kroneckerfactors(inds))))[i′]
    return inds[i_linear]
end

function Base.checkindex(::Type{Bool}, inds::CartesianProductUnitRange, i::CartesianPair)
    indsa, indsb = kroneckerfactors(inds)
    ia, ib = kroneckerfactors(i)
    return checkindex(Bool, indsa, ia) && checkindex(Bool, indsb, ib)
end


# AbstractUnitRange interface
# ---------------------------
Base.first(r::CartesianProductUnitRange) = first(r.range)
Base.last(r::CartesianProductUnitRange) = last(r.range)


# Broadcasting
# ------------
for f in (:+, :-)
    @eval BC.broadcasted(::BC.DefaultArrayStyle{1}, ::typeof($f), r::CartesianProductUnitRange, x::Integer) =
        cartesianrange(kroneckerfactors(r)..., $f.(unproduct(r), x))
    @eval BC.broadcasted(::BC.DefaultArrayStyle{1}, ::typeof($f), x::Integer, r::CartesianProductUnitRange) =
        cartesianrange(kroneckerfactors(r)..., $f.(x, unproduct(r)))
end

function BC.axistype(r1::CartesianProductUnitRange, r2::CartesianProductUnitRange)
    r1a, r1b = kroneckerfactors(r1)
    r2a, r2b = kroneckerfactors(r2)
    return cartesianrange(splat(BC.axistype).(((r1a, r2a), (r1b, r2b), (unproduct(r1), unproduct(r2))))...)
end


# Show
# ----
function Base.show(io::IO, ab::Union{CartesianPair, CartesianProduct})
    a, b = kroneckerfactors(ab)
    show(io, a)
    print(io, " × ")
    show(io, b)
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", ab::Union{CartesianPair, CartesianProduct})
    a, b = kroneckerfactors(ab)
    compact = get(io, :compact, true)::Bool
    show(io, mime, a)
    compact || println(io)
    print(io, " × ")
    compact || println(io)
    show(io, mime, b)
    return nothing
end

function Base.show(io::IO, ab::CartesianProductVector)
    a, b = kroneckerfactors(ab)
    print(io, "cartesianproduct(")
    show(io, a)
    print(io, ", ")
    show(io, b)
    print(io, ", ")
    show(io, unproduct(ab))
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ab::CartesianProductUnitRange)
    a, b = kroneckerfactors(ab)
    range = unproduct(ab)
    print(io, "cartesianrange(")
    show(io, a)
    print(io, ", ")
    show(io, b)
    print(io, ", ")
    show(io, unproduct(ab))
    print(io, ")")
    return nothing
end
function Base.show(io::IO, ab::CartesianProductOneTo)
    a, b = kroneckerfactors(ab)
    print(io, "(")
    show(io, a)
    print(io, " × ")
    show(io, b)
    print(io, ")")
    return nothing
end
