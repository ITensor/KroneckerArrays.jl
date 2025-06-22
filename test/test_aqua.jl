using BlockSparseArrays: BlockSparseArrays
using KroneckerArrays: KroneckerArrays
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
  # TODO: Investigate and fix ambiguities.
  Aqua.test_all(KroneckerArrays; ambiguities=false)
end
