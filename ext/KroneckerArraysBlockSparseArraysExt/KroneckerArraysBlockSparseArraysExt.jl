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
using BlockSparseArrays: Block, GetUnstoredBlock, eachblockaxis, mortar_axis
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
  return mortar_axis(arg2.(eachblockaxis(r)))
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

function (f::GetUnstoredBlock)(
  ::Type{<:AbstractMatrix{KroneckerMatrix{T,A,B}}}, I::Vararg{Int,2}
) where {T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}}
  ax_a = arg1.(f.axes)
  f_a = GetUnstoredBlock(ax_a)
  a = f_a(AbstractMatrix{A}, I...)

  ax_b = arg2.(f.axes)
  f_b = GetUnstoredBlock(ax_b)
  b = f_b(AbstractMatrix{B}, I...)

  return a ⊗ b
end
function (f::GetUnstoredBlock)(
  ::Type{<:AbstractMatrix{EyeKronecker{T,A,B}}}, I::Vararg{Int,2}
) where {T,A<:Eye{T},B<:AbstractMatrix{T}}
  block_ax_a = arg1.(block_axes(f.axes, Block(I)))
  a = _similar(A, block_ax_a)

  ax_b = arg2.(f.axes)
  f_b = GetUnstoredBlock(ax_b)
  b = f_b(AbstractMatrix{B}, I...)

  return a ⊗ b
end
function (f::GetUnstoredBlock)(
  ::Type{<:AbstractMatrix{KroneckerEye{T,A,B}}}, I::Vararg{Int,2}
) where {T,A<:AbstractMatrix{T},B<:Eye{T}}
  ax_a = arg1.(f.axes)
  f_a = GetUnstoredBlock(ax_a)
  a = f_a(AbstractMatrix{A}, I...)

  block_ax_b = arg2.(block_axes(f.axes, Block(I)))
  b = _similar(B, block_ax_b)

  return a ⊗ b
end
function (f::GetUnstoredBlock)(
  ::Type{<:AbstractMatrix{EyeEye{T,A,B}}}, I::Vararg{Int,2}
) where {T,A<:Eye{T},B<:Eye{T}}
  return error("Not implemented.")
end

end
