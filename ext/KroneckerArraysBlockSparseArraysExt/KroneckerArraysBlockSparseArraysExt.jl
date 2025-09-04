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
using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗, arg1, arg2, _similar
using BlockSparseArrays.TypeParameterAccessors: unwrap_array_type

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
  a::ZeroBlocks{N,KroneckerArray{T,N,A,B}}, I::Vararg{Int,N}
) where {T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}}
  ax_a1 = map(arg1, a.parentaxes)
  ax_a2 = map(arg2, a.parentaxes)
  # TODO: Instead of mutability, maybe have a trait like
  # `isstructural` or `isdata`.
  ismut1 = ismutabletype(unwrap_array_type(A))
  ismut2 = ismutabletype(unwrap_array_type(B))
  (ismut1 || ismut2) || error("Can't get zero block.")
  a1 = if ismut1
    ZeroBlocks{N,A}(ax_a1)[I...]
  else
    block_ax_a1 = arg1.(block_axes(a.parentaxes, Block(I)))
    _similar(A, block_ax_a1)
  end
  a2 = if ismut2
    ZeroBlocks{N,B}(ax_a2)[I...]
  else
    block_ax_a2 = arg2.(block_axes(a.parentaxes, Block(I)))
    a2 = _similar(B, block_ax_a2)
  end
  return a1 ⊗ a2
end

using BlockSparseArrays: BlockSparseArrays
using KroneckerArrays: KroneckerArrays, KroneckerVector
function BlockSparseArrays.to_truncated_indices(values::KroneckerVector, I)
  return KroneckerArrays.to_truncated_indices(values, I)
end

end
