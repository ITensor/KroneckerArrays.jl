using FillArrays: FillArrays, Ones, Zeros
function FillArrays.fillsimilar(
        a::Zeros{T},
        ax::Tuple{
            CartesianProductUnitRange{<:Integer}, Vararg{CartesianProductUnitRange{<:Integer}},
        }
    ) where {T}
    return Zeros{T}(kroneckerfactors.(ax, 1)) ⊗ Zeros{T}(kroneckerfactors.(ax, 2))
end

# Simplification rules similar to those for FillArrays.jl:
# https://github.com/JuliaArrays/FillArrays.jl/blob/v1.13.0/src/fillbroadcast.jl
using FillArrays: Zeros
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(+),
        a::KroneckerArray,
        b::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros}
    )
    # TODO: Promote the element types.
    return a
end
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(+),
        a::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros},
        b::KroneckerArray
    )
    # TODO: Promote the element types.
    return b
end
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(+),
        a::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros},
        b::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros}
    )
    # TODO: Promote the element types and axes.
    return b
end
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(-),
        a::KroneckerArray,
        b::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros}
    )
    # TODO: Promote the element types.
    return a
end
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(-),
        a::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros},
        b::KroneckerArray
    )
    # TODO: Promote the element types.
    # TODO: Return `broadcasted(-, b)`.
    return -b
end
function Base.broadcasted(
        style::KroneckerStyle,
        ::typeof(-),
        a::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros},
        b::KroneckerArray{<:Any, <:Any, <:Zeros, <:Zeros}
    )
    # TODO: Promote the element types and axes.
    return b
end
