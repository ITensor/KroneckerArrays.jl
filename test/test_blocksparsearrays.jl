using Adapt: adapt
using BlockArrays: Block, BlockRange
using BlockSparseArrays:
  BlockSparseArray, BlockSparseMatrix, blockrange, blocksparse, blocktype
using FillArrays: Eye, SquareEye
using JLArrays: JLArray
using KroneckerArrays: KroneckerArray, ⊗, ×
using LinearAlgebra: norm
using MatrixAlgebraKit: svd_compact
using Test: @test, @test_broken, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "BlockSparseArraysExt, KroneckerArray blocks (arraytype=$arrayt, eltype=$elt)" for arrayt in
                                                                                            arrayts,
  elt in elts

  dev = adapt(arrayt)
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(randn(elt, 2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(randn(elt, 3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, r, r))
  @test blocktype(a) === valtype(d)
  @test a isa BlockSparseMatrix{elt,valtype(d)}
  @test a[Block(1, 1)] == dev(d[Block(1, 1)])
  @test a[Block(1, 1)] isa valtype(d)
  @test a[Block(2, 2)] == dev(d[Block(2, 2)])
  @test a[Block(2, 2)] isa valtype(d)
  @test iszero(a[Block(2, 1)])
  @test a[Block(2, 1)] == dev(zeros(elt, 3, 2) ⊗ zeros(elt, 3, 2))
  @test a[Block(2, 1)] isa valtype(d)
  @test iszero(a[Block(1, 2)])
  @test a[Block(1, 2)] == dev(zeros(elt, 2, 3) ⊗ zeros(elt, 2, 3))
  @test a[Block(1, 2)] isa valtype(d)

  b = a * a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) * Array(a)

  b = a + a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) + Array(a)

  b = 3a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ 3Array(a)

  b = a / 3
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) / 3

  @test norm(a) ≈ norm(Array(a))

  if arrayt == Array
    @test Array(inv(a)) ≈ inv(Array(a))
  else
    # Broken for JLArray, it seems like `inv` isn't
    # type stable.
    @test_broken inv(a)
  end

  # Broken operations
  @test_broken exp(a)
  @test_broken svd_compact(a)
  @test_broken a[Block.(1:2), Block(2)]
end

@testset "BlockSparseArraysExt, SquareEyeKronecker blocks (arraytype=$arrayt, eltype=$elt)" for arrayt in
                                                                                                (
    Array,
  ),
  elt in elts

  dev = adapt(arrayt)
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => Eye{elt}(2) ⊗ randn(elt, 2, 2),
    Block(2, 2) => Eye{elt}(3) ⊗ randn(elt, 3, 3),
  )
  a = dev(blocksparse(d, r, r))
  @test blocktype(a) === valtype(d)
  @test a isa BlockSparseMatrix{elt,valtype(d)}
  @test a[Block(1, 1)] == dev(d[Block(1, 1)])
  @test a[Block(1, 1)] isa valtype(d)
  @test a[Block(2, 2)] == dev(d[Block(2, 2)])
  @test a[Block(2, 2)] isa valtype(d)
  @test_broken iszero(a[Block(2, 1)])
  @test_broken a[Block(2, 1)] == dev(zeros(elt, 3, 2) ⊗ zeros(elt, 3, 2))
  @test_broken a[Block(2, 1)] isa valtype(d)
  @test_broken iszero(a[Block(1, 2)])
  @test_broken a[Block(1, 2)] == dev(zeros(elt, 2, 3) ⊗ zeros(elt, 2, 3))
  @test_broken a[Block(1, 2)] isa valtype(d)

  b = a * a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) * Array(a)

  b = a + a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) + Array(a)

  b = 3a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ 3Array(a)

  b = a / 3
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) / 3

  @test norm(a) ≈ norm(Array(a))

  if arrayt == Array
    @test Array(inv(a)) ≈ inv(Array(a))
  else
    # Broken for JLArray, it seems like `inv` isn't
    # type stable.
    @test_broken inv(a)
  end

  # Broken operations
  # @test_broken exp(a)
  @test_broken svd_compact(a)
  @test_broken a[Block.(1:2), Block(2)]
end
