using Adapt: adapt
using BlockArrays: Block, BlockRange, blockedrange, blockisequal, mortar
using BlockSparseArrays:
  BlockIndexVector,
  BlockSparseArray,
  BlockSparseMatrix,
  blockrange,
  blocksparse,
  blocktype,
  eachblockaxis
using FillArrays: Eye, SquareEye
using JLArrays: JLArray
using KroneckerArrays: KroneckerArray, ⊗, ×, arg1, arg2, cartesianrange
using LinearAlgebra: norm
using MatrixAlgebraKit: svd_compact, svd_trunc
using StableRNGs: StableRNG
using Test: @test, @test_broken, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, ComplexF32)
arrayts = (Array, JLArray)
@testset "BlockSparseArraysExt, KroneckerArray blocks (arraytype=$arrayt, eltype=$elt)" for arrayt in
                                                                                            arrayts,
  elt in elts

  # BlockUnitRange with CartesianProduct blocks
  r = blockrange([2 × 3, 3 × 4])
  @test r[Block(1)] ≡ cartesianrange(2 × 3, 1:6)
  @test r[Block(2)] ≡ cartesianrange(3 × 4, 7:18)
  @test eachblockaxis(r)[1] ≡ cartesianrange(2, 3)
  @test eachblockaxis(r)[2] ≡ cartesianrange(3, 4)
  @test blockisequal(arg1(r), blockedrange([2, 3]))
  @test blockisequal(arg2(r), blockedrange([3, 4]))

  r = blockrange([2 × 3, 3 × 4])
  r′ = r[Block.([2, 1])]
  @test r′[Block(1)] ≡ cartesianrange(3 × 4, 7:18)
  @test r′[Block(2)] ≡ cartesianrange(2 × 3, 1:6)
  @test eachblockaxis(r′)[1] ≡ cartesianrange(3, 4)
  @test eachblockaxis(r′)[2] ≡ cartesianrange(2, 3)

  dev = adapt(arrayt)
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(randn(elt, 2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(randn(elt, 3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  @test sprint(show, a) isa String
  @test sprint(show, MIME("text/plain"), a) isa String
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

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(randn(elt, 2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(randn(elt, 3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  @test a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]] ==
    a[Block(2, 2)][(2:3) × (2:3), (2:3) × (2:3)]
  @test a[Block(2, 2)[(:) × (2:3), (:) × (2:3)]] == a[Block(2, 2)][(:) × (2:3), (:) × (2:3)]
  @test a[Block(2, 2)[(1:2) × (2:3), (:) × (2:3)]] ==
    a[Block(2, 2)][(1:2) × (2:3), (:) × (2:3)]

  # Blockwise slicing, shows up in truncated block sparse matrix factorizations.
  I1 = BlockIndexVector(Block(1), Base.Slice(Base.OneTo(2)) × [1])
  I2 = BlockIndexVector(Block(2), Base.Slice(Base.OneTo(3)) × [1, 3])
  I = [I1, I2]
  b = a[I, I]
  @test b[Block(1, 1)] == a[Block(1, 1)[(1:2) × [1], (1:2) × [1]]]
  @test iszero(b[Block(2, 1)])
  @test iszero(b[Block(1, 2)])
  @test b[Block(2, 2)] == a[Block(2, 2)[(1:3) × [1, 3], (1:3) × [1, 3]]]

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(randn(elt, 2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(randn(elt, 3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  i1 = Block(1)[(1:2) × (1:2)]
  i2 = Block(2)[(2:3) × (2:3)]
  I = mortar([i1, i2])
  b = @view a[I, I]
  b′ = copy(b)
  @test b[Block(2, 2)] == b′[Block(2, 2)] == a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]]
  @test_broken b[Block(1, 2)]

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(randn(elt, 2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(randn(elt, 3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  i1 = Block(1)[(1:2) × (1:2)]
  i2 = Block(2)[(2:3) × (2:3)]
  I = [i1, i2]
  b = @view a[I, I]
  b′ = copy(b)
  @test b[Block(2, 2)] == b′[Block(2, 2)] == a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]]
  @test_broken b[Block(1, 2)]

  # Matrix multiplication
  b = a * a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) * Array(a)

  # Addition (mapping, broadcasting)
  b = a + a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) + Array(a)

  # Scaling (mapping, broadcasting)
  b = 3a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ 3Array(a)

  # Dividing (mapping, broadcasting)
  b = a / 3
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) / 3

  # Norm
  @test norm(a) ≈ norm(Array(a))

  if arrayt === Array
    @test Array(inv(a)) ≈ inv(Array(a))
  else
    # Broken on GPU.
    @test_broken inv(a)
  end

  u, s, v = svd_compact(a)
  @test Array(u * s * v) ≈ Array(a)

  b = a[Block.(1:2), Block(2)]
  @test b[Block(1)] == a[Block(1, 2)]
  @test b[Block(2)] == a[Block(2, 2)]

  # Broken operations
  @test_broken exp(a)
end

@testset "BlockSparseArraysExt, EyeKronecker blocks (arraytype=$arrayt, eltype=$elt)" for arrayt in
                                                                                          arrayts,
  elt in elts

  dev = adapt(arrayt)
  r = @constinferred blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => Eye{elt}(2, 2) ⊗ dev(randn(elt, 2, 2)),
    Block(2, 2) => Eye{elt}(3, 3) ⊗ dev(randn(elt, 3, 3)),
  )
  a = @constinferred dev(blocksparse(d, (r, r)))
  @test sprint(show, a) == sprint(show, Array(a))
  @test sprint(show, MIME("text/plain"), a) isa String
  @test @constinferred(blocktype(a)) === valtype(d)
  @test a isa BlockSparseMatrix{elt,valtype(d)}
  @test @constinferred(a[Block(1, 1)]) == dev(d[Block(1, 1)])
  @test @constinferred(a[Block(1, 1)]) isa valtype(d)
  @test @constinferred(a[Block(2, 2)]) == dev(d[Block(2, 2)])
  @test @constinferred(a[Block(2, 2)]) isa valtype(d)
  @test @constinferred(iszero(a[Block(2, 1)]))
  @test a[Block(2, 1)] == dev(Eye(3, 2) ⊗ zeros(elt, 3, 2))
  @test a[Block(2, 1)] isa valtype(d)
  @test iszero(a[Block(1, 2)])
  @test a[Block(1, 2)] == dev(Eye(2, 3) ⊗ zeros(elt, 2, 3))
  @test a[Block(1, 2)] isa valtype(d)

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  @test a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]] ==
    a[Block(2, 2)][(2:3) × (2:3), (2:3) × (2:3)]
  @test a[Block(2, 2)[(:) × (2:3), (:) × (2:3)]] == a[Block(2, 2)][(:) × (2:3), (:) × (2:3)]
  @test a[Block(2, 2)[(1:2) × (2:3), (:) × (2:3)]] ==
    a[Block(2, 2)][(1:2) × (2:3), (:) × (2:3)]

  # Blockwise slicing, shows up in truncated block sparse matrix factorizations.
  I1 = BlockIndexVector(Block(1), Base.Slice(Base.OneTo(2)) × [1])
  I2 = BlockIndexVector(Block(2), Base.Slice(Base.OneTo(3)) × [1, 3])
  I = [I1, I2]
  b = a[I, I]
  @test b[Block(1, 1)] == a[Block(1, 1)[(1:2) × [1], (1:2) × [1]]]
  @test arg1(b[Block(1, 1)]) isa Eye
  @test iszero(b[Block(2, 1)])
  @test arg1(b[Block(2, 1)]) isa Eye
  @test iszero(b[Block(1, 2)])
  @test arg1(b[Block(1, 2)]) isa Eye
  @test b[Block(2, 2)] == a[Block(2, 2)[(1:3) × [1, 3], (1:3) × [1, 3]]]
  @test arg1(b[Block(2, 2)]) isa Eye

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  i1 = Block(1)[(1:2) × (1:2)]
  i2 = Block(2)[(2:3) × (2:3)]
  I = mortar([i1, i2])
  b = @view a[I, I]
  @test b[Block(2, 2)] == a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]]
  @test_broken copy(b)
  @test_broken b[Block(1, 2)]

  # Slicing
  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  i1 = Block(1)[(1:2) × (1:2)]
  i2 = Block(2)[(2:3) × (2:3)]
  I = [i1, i2]
  b = @view a[I, I]
  @test b[Block(2, 2)] == a[Block(2, 2)[(2:3) × (2:3), (2:3) × (2:3)]]
  @test_broken copy(b)
  @test_broken b[Block(1, 2)]

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  b = @constinferred a * a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) * Array(a)

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  # Type inference is broken for this operation.
  # b = @constinferred a + a
  b = a + a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) + Array(a)

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  # Type inference is broken for this operation.
  # b = @constinferred 3a
  b = 3a
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ 3Array(a)

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  # Type inference is broken for this operation.
  # b = @constinferred a / 3
  b = a / 3
  @test typeof(b) === typeof(a)
  @test Array(b) ≈ Array(a) / 3

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  @test @constinferred(norm(a)) ≈ norm(Array(a))

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  if arrayt === Array
    b = @constinferred exp(a)
    @test Array(b) ≈ exp(Array(a))
  else
    @test_broken exp(a)
  end

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  u, s, v = svd_compact(a)
  @test u * s * v ≈ a
  @test blocktype(u) >: blocktype(u)
  @test eltype(u) === eltype(a)
  @test blocktype(v) >: blocktype(a)
  @test eltype(v) === eltype(a)
  @test eltype(s) === real(eltype(a))

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  if arrayt === Array
    @test Array(inv(a)) ≈ inv(Array(a))
  else
    # Broken on GPU.
    @test_broken inv(a)
  end

  r = blockrange([2 × 2, 3 × 3])
  d = Dict(
    Block(1, 1) => dev(Eye{elt}(2, 2) ⊗ randn(elt, 2, 2)),
    Block(2, 2) => dev(Eye{elt}(3, 3) ⊗ randn(elt, 3, 3)),
  )
  a = dev(blocksparse(d, (r, r)))
  # Broken operations
  b = a[Block.(1:2), Block(2)]
  @test b[Block(1)] == a[Block(1, 2)]
  @test b[Block(2)] == a[Block(2, 2)]

  # svd_trunc
  dev = adapt(arrayt)
  r = @constinferred blockrange([2 × 2, 3 × 3])
  rng = StableRNG(1234)
  d = Dict(
    Block(1, 1) => Eye{elt}(2, 2) ⊗ randn(rng, elt, 2, 2),
    Block(2, 2) => Eye{elt}(3, 3) ⊗ randn(rng, elt, 3, 3),
  )
  a = @constinferred dev(blocksparse(d, (r, r)))
  if arrayt === Array
    u, s, v = svd_trunc(a; trunc=(; maxrank=6))
    u′, s′, v′ = svd_trunc(Matrix(a); trunc=(; maxrank=5))
    @test Matrix(u * s * v) ≈ u′ * s′ * v′
  else
    @test_broken svd_trunc(a; trunc=(; maxrank=6))
  end

  @testset "Block deficient" begin
    da = Dict(Block(1, 1) => Eye{elt}(2, 2) ⊗ dev(randn(elt, 2, 2)))
    a = @constinferred dev(blocksparse(da, (r, r)))

    db = Dict(Block(2, 2) => Eye{elt}(3, 3) ⊗ dev(randn(elt, 3, 3)))
    b = @constinferred dev(blocksparse(db, (r, r)))

    @test Array(a + b) ≈ Array(a) + Array(b)
    @test Array(2a) ≈ 2Array(a)
  end
end
