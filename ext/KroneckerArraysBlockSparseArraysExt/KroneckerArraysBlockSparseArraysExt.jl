module KroneckerArraysBlockSparseArraysExt

using BlockArrays: Block
using BlockSparseArrays: BlockIndexVector, GenericBlockIndex
using KroneckerArrays: CartesianPair, CartesianProduct
function Base.getindex(
  b::Block,
  I1::Union{CartesianPair,CartesianProduct},
  Irest::Union{CartesianPair,CartesianProduct}...,
)
  return GenericBlockIndex(b, (I1, Irest...))
end
function Base.getindex(b::Block, I1::CartesianProduct, Irest::CartesianProduct...)
  return BlockIndexVector(b, (I1, Irest...))
end

using BlockSparseArrays: BlockSparseArrays, blockrange
using KroneckerArrays: CartesianPair, CartesianProduct, cartesianrange
function BlockSparseArrays.blockrange(bs::Vector{<:CartesianPair})
  return blockrange(map(cartesianrange, bs))
end
function BlockSparseArrays.blockrange(bs::Vector{<:CartesianProduct})
  return blockrange(map(cartesianrange, bs))
end

using BlockArrays: BlockArrays, mortar
using BlockSparseArrays: blockrange
using KroneckerArrays: CartesianProductUnitRange
# Makes sure that `mortar` results in a `BlockVector` with the correct
# axes, otherwise the axes would not preserve the Kronecker structure.
# This is helpful when indexing `BlockUnitRange`, for example:
# https://github.com/JuliaArrays/BlockArrays.jl/blob/v1.7.1/src/blockaxis.jl#L540-L547
function BlockArrays.mortar(blocks::AbstractVector{<:CartesianProductUnitRange})
  return mortar(blocks, (blockrange(map(Base.axes1, blocks)),))
end

using BlockArrays: AbstractBlockedUnitRange
using BlockSparseArrays: Block, ZeroBlocks, eachblockaxis, mortar_axis
using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗, arg1, arg2, isactive

function KroneckerArrays.arg1(r::AbstractBlockedUnitRange)
  return mortar_axis(arg1.(eachblockaxis(r)))
end
function KroneckerArrays.arg2(r::AbstractBlockedUnitRange)
  return mortar_axis(arg2.(eachblockaxis(r)))
end

function block_axes(
  ax::NTuple{N,AbstractUnitRange{<:Integer}}, I::Vararg{Block{1},N}
) where {N}
  return ntuple(N) do d
    return only(axes(ax[d][I[d]]))
  end
end
function block_axes(ax::NTuple{N,AbstractUnitRange{<:Integer}}, I::Block{N}) where {N}
  return block_axes(ax, Tuple(I)...)
end

## TODO: Is this needed?
function Base.getindex(
  a::ZeroBlocks{N,KroneckerArray{T,N,A1,A2}}, I::Vararg{Int,N}
) where {T,N,A1<:AbstractArray{T,N},A2<:AbstractArray{T,N}}
  ax_a1 = map(arg1, a.parentaxes)
  ax_a2 = map(arg2, a.parentaxes)
  block_ax_a1 = arg1.(block_axes(a.parentaxes, Block(I)))
  block_ax_a2 = arg2.(block_axes(a.parentaxes, Block(I)))
  # TODO: Is this a good definition? It is similar to
  # the definition of `similar` and `adapt_structure`.
  return if isactive(A1) == isactive(A2)
    ZeroBlocks{N,A1}(ax_a1)[I...] ⊗ ZeroBlocks{N,A2}(ax_a2)[I...]
  elseif isactive(A1)
    ZeroBlocks{N,A1}(ax_a1)[I...] ⊗ A2(block_ax_a2)
  elseif isactive(A2)
    A1(block_ax_a1) ⊗ ZeroBlocks{N,A2}(ax_a2)[I...]
  end
end

using BlockSparseArrays: BlockSparseArrays
using KroneckerArrays: KroneckerArrays, KroneckerVector
function BlockSparseArrays.to_truncated_indices(values::KroneckerVector, I)
  return KroneckerArrays.to_truncated_indices(values, I)
end

end
