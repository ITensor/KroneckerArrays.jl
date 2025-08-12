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

using BlockArrays: AbstractBlockedUnitRange
using BlockSparseArrays: Block, ZeroBlocks, eachblockaxis, mortar_axis
using DerivableInterfaces: zero!
using FillArrays: Eye
using KroneckerArrays:
  KroneckerArrays,
  EyeEye,
  EyeKronecker,
  KroneckerEye,
  KroneckerMatrix,
  ⊗,
  arg1,
  arg2,
  _similar

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
  a::ZeroBlocks{2,KroneckerMatrix{T,A,B}}, I::Vararg{Int,2}
) where {T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}}
  ax_a1 = map(arg1, a.parentaxes)
  a1 = ZeroBlocks{2,A}(ax_a1)[I...]
  ax_a2 = map(arg2, a.parentaxes)
  a2 = ZeroBlocks{2,B}(ax_a2)[I...]
  return a1 ⊗ a2
end
function Base.getindex(
  a::ZeroBlocks{2,EyeKronecker{T,A,B}}, I::Vararg{Int,2}
) where {T,A<:Eye{T},B<:AbstractMatrix{T}}
  block_ax_a1 = arg1.(block_axes(a.parentaxes, Block(I)))
  a1 = _similar(A, block_ax_a1)

  ax_a2 = arg2.(a.parentaxes)
  a2 = ZeroBlocks{2,B}(ax_a2)[I...]

  return a1 ⊗ a2
end
function Base.getindex(
  a::ZeroBlocks{2,KroneckerEye{T,A,B}}, I::Vararg{Int,2}
) where {T,A<:AbstractMatrix{T},B<:Eye{T}}
  ax_a1 = arg1.(a.parentaxes)
  a1 = ZeroBlocks{2,A}(ax_a1)[I...]

  block_ax_a2 = arg2.(block_axes(a.parentaxes, Block(I)))
  a2 = _similar(B, block_ax_a2)

  return a1 ⊗ a2
end
function Base.getindex(
  a::ZeroBlocks{2,EyeEye{T,A,B}}, I::Vararg{Int,2}
) where {T,A<:Eye{T},B<:Eye{T}}
  return error("Not implemented.")
end

using BlockSparseArrays: BlockSparseArrays
using KroneckerArrays: KroneckerArrays, KroneckerVector
function BlockSparseArrays.to_truncated_indices(values::KroneckerVector, I)
  return KroneckerArrays.to_truncated_indices(values, I)
end

end
