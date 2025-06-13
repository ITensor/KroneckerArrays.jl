module KroneckerArraysBlockSparseArraysExt

using BlockSparseArrays: BlockSparseArrays, blockrange
using KroneckerArrays: CartesianProduct, cartesianrange

function BlockSparseArrays.blockrange(bs::Vector{<:CartesianProduct})
  return blockrange(map(cartesianrange, bs))
end

end
