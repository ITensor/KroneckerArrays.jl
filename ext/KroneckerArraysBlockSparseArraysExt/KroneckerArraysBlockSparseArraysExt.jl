module KroneckerArraysBlockSparseArraysExt

using BlockArrays: BlockArrays, AbstractBlockedUnitRange, Block, mortar
using BlockSparseArrays: BlockSparseArrays, BlockIndexVector, GenericBlockIndex, ZeroBlocks,
    blockrange, eachblockaxis, mortar_axis
using DiagonalArrays: ShapeInitializer
using KroneckerArrays: KroneckerArrays, CartesianPair, CartesianProduct,
    CartesianProductUnitRange, KroneckerArray, KroneckerVector, cartesianrange, isactive,
    kroneckerfactors, ⊗

function Base.getindex(
        b::Block{N},
        I::Vararg{Union{CartesianPair, CartesianProduct}, N}
    ) where {N}
    return GenericBlockIndex(b, I)
end
function Base.getindex(b::Block{N}, I::Vararg{CartesianProduct, N}) where {N}
    return BlockIndexVector(b, I)
end

function BlockSparseArrays.blockrange(bs::Vector{<:CartesianPair})
    return blockrange(map(cartesianrange, bs))
end
function BlockSparseArrays.blockrange(bs::Vector{<:CartesianProduct})
    return blockrange(map(cartesianrange, bs))
end

# Makes sure that `mortar` results in a `BlockVector` with the correct
# axes, otherwise the axes would not preserve the Kronecker structure.
# This is helpful when indexing `BlockUnitRange`, for example:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.7.1/src/blockaxis.jl#L540-L547
function BlockArrays.mortar(blocks::AbstractVector{<:CartesianProductUnitRange})
    return mortar(blocks, (blockrange(map(Base.axes1, blocks)),))
end

function KroneckerArrays.kroneckerfactors(r::AbstractBlockedUnitRange, i::Int)
    return mortar_axis(kroneckerfactors.(eachblockaxis(r), i))
end
function KroneckerArrays.kroneckerfactors(r::AbstractBlockedUnitRange)
    return (kroneckerfactors(r, 1), kroneckerfactors(r, 2))
end

function block_axes(
        ax::NTuple{N, AbstractUnitRange{<:Integer}},
        I::Vararg{Block{1}, N}
    ) where {N}
    return ntuple(N) do d
        return only(axes(ax[d][I[d]]))
    end
end
function block_axes(ax::NTuple{N, AbstractUnitRange{<:Integer}}, I::Block{N}) where {N}
    return block_axes(ax, Tuple(I)...)
end

## TODO: Is this needed?
function Base.getindex(
        a::ZeroBlocks{N, KroneckerArray{T, N, A1, A2}}, I::Vararg{Int, N}
    ) where {T, N, A1 <: AbstractArray{T, N}, A2 <: AbstractArray{T, N}}
    ax_a1 = kroneckerfactors.(a.parentaxes, 1)
    ax_a2 = kroneckerfactors.(a.parentaxes, 2)
    block_ax_a1 = kroneckerfactors.(block_axes(a.parentaxes, Block(I)), 1)
    block_ax_a2 = kroneckerfactors.(block_axes(a.parentaxes, Block(I)), 2)
    # TODO: Is this a good definition? It is similar to
    # the definition of `similar` and `adapt_structure`.
    return if isactive(A1) == isactive(A2)
        ZeroBlocks{N, A1}(ax_a1)[I...] ⊗ ZeroBlocks{N, A2}(ax_a2)[I...]
    elseif isactive(A1)
        ZeroBlocks{N, A1}(ax_a1)[I...] ⊗ A2(ShapeInitializer(), block_ax_a2)
    elseif isactive(A2)
        A1(ShapeInitializer(), block_ax_a1) ⊗ ZeroBlocks{N, A2}(ax_a2)[I...]
    end
end

function BlockSparseArrays.to_truncated_indices(values::KroneckerVector, I)
    return KroneckerArrays.to_truncated_indices(values, I)
end

end
