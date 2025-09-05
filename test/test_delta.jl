using Adapt: adapt
using DerivableInterfaces: zero!
using DiagonalArrays: δ
using FillArrays: Eye, Zeros
using JLArrays: JLArray, jl
using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗, ×, arg1, arg2, cartesianrange
using LinearAlgebra: det, norm, pinv
using StableRNGs: StableRNG
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred

@testset "FillArrays.Eye, DiagonalArrays.Delta" begin
  MATRIX_FUNCTIONS = KroneckerArrays.MATRIX_FUNCTIONS
  if VERSION < v"1.11-"
    # `cbrt(::AbstractMatrix{<:Real})` was implemented in Julia 1.11.
    MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, [:cbrt])
  end

  a = Eye(2) ⊗ randn(3, 3)
  @test size(a) == (6, 6)
  @test a + a == Eye(2) ⊗ (2 * arg2(a))
  @test 2a == Eye(2) ⊗ (2 * arg2(a))
  @test a * a == Eye(2) ⊗ (arg2(a) * arg2(a))
  @test_broken arg1(a[(:) × (:), (:) × (:)]) ≡ Eye(2)
  @test_broken arg1(view(a, (:) × (:), (:) × (:))) ≡ Eye(2)
  @test_broken arg1(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)]) ≡ Eye(2)
  @test_broken arg1(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:))) ≡ Eye(2)
  @test_broken arg1(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡ Eye(2)
  @test_broken arg1(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:))) ≡ Eye(2)
  @test_broken arg1(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡
    Eye(2)
  @test_broken arg1(
    view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:))
  ) ≡ Eye(2)
  @test arg1(adapt(JLArray, a)) ≡ Eye(2)
  @test arg2(adapt(JLArray, a)) == jl(arg2(a))
  @test arg2(adapt(JLArray, a)) isa JLArray
  @test_broken arg1(similar(a, (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡ Eye(3)
  @test_broken arg1(similar(typeof(a), (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡
    Eye(3)
  @test_broken arg1(similar(a, Float32, (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡
    Eye{Float32}(3)
  @test arg1(copy(a)) ≡ Eye(2)
  @test arg2(copy(a)) == arg2(a)
  b = similar(a)
  @test arg1(copyto!(b, a)) ≡ Eye(2)
  @test arg2(copyto!(b, a)) == arg2(a)
  @test arg1(permutedims(a, (2, 1))) ≡ Eye(2)
  @test arg2(permutedims(a, (2, 1))) == permutedims(arg2(a), (2, 1))
  b = similar(a)
  @test arg1(permutedims!(b, a, (2, 1))) ≡ Eye(2)
  @test arg2(permutedims!(b, a, (2, 1))) == permutedims(arg2(a), (2, 1))

  a = randn(3, 3) ⊗ Eye(2)
  @test size(a) == (6, 6)
  @test a + a == (2 * arg1(a)) ⊗ Eye(2)
  @test 2a == (2 * arg1(a)) ⊗ Eye(2)
  @test a * a == (arg1(a) * arg1(a)) ⊗ Eye(2)
  @test_broken arg2(a[(:) × (:), (:) × (:)]) ≡ Eye(2)
  @test_broken arg2(view(a, (:) × (:), (:) × (:))) ≡ Eye(2)
  @test_broken arg2(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)]) ≡ Eye(2)
  @test_broken arg2(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:))) ≡ Eye(2)
  @test_broken arg2(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡ Eye(2)
  @test_broken arg2(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:))) ≡ Eye(2)
  @test_broken arg2(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡
    Eye(2)
  @test_broken arg2(
    view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:))
  ) ≡ Eye(2)
  @test arg2(adapt(JLArray, a)) ≡ Eye(2)
  @test arg1(adapt(JLArray, a)) == jl(arg1(a))
  @test arg1(adapt(JLArray, a)) isa JLArray
  @test_broken arg2(similar(a, (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡ Eye(3)
  @test_broken arg2(similar(typeof(a), (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡
    Eye(3)
  @test_broken arg2(similar(a, Float32, (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡
    Eye{Float32}(3)
  @test arg2(copy(a)) ≡ Eye(2)
  @test arg2(copy(a)) == arg2(a)
  b = similar(a)
  @test arg2(copyto!(b, a)) ≡ Eye(2)
  @test arg2(copyto!(b, a)) == arg2(a)
  @test arg2(permutedims(a, (2, 1))) ≡ Eye(2)
  @test arg1(permutedims(a, (2, 1))) == permutedims(arg1(a), (2, 1))
  b = similar(a)
  @test arg2(permutedims!(b, a, (2, 1))) ≡ Eye(2)
  @test arg1(permutedims!(b, a, (2, 1))) == permutedims(arg1(a), (2, 1))

  a = δ(2, 2) ⊗ randn(3, 3)
  @test size(a) == (6, 6)
  @test a + a == δ(2, 2) ⊗ (2 * arg2(a))
  @test 2a == δ(2, 2) ⊗ (2 * arg2(a))
  @test a * a == δ(2, 2) ⊗ (arg2(a) * arg2(a))
  @test_broken arg1(a[(:) × (:), (:) × (:)]) ≡ δ(2, 2)
  @test_broken arg1(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)]) ≡ δ(2, 2)
  @test_broken arg1(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:))) ≡ δ(2, 2)
  @test_broken arg1(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡ δ(2, 2)
  @test_broken arg1(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:))) ≡ δ(2, 2)
  @test_broken arg1(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡
    δ(2, 2)
  @test_broken arg1(
    view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:))
  ) ≡ δ(2, 2)
  @test arg1(adapt(JLArray, a)) ≡ δ(2, 2)
  @test arg2(adapt(JLArray, a)) == jl(arg2(a))
  @test arg2(adapt(JLArray, a)) isa JLArray
  @test_broken arg1(similar(a, (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡ δ(3, 3)
  @test_broken arg1(similar(typeof(a), (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡
    δ(3, 3)
  @test_broken arg1(similar(a, Float32, (cartesianrange(3 × 2), cartesianrange(3 × 2)))) ≡
    δ(Float32, 3, 3)
  @test arg1(copy(a)) ≡ δ(2, 2)
  @test arg2(copy(a)) == arg2(a)
  b = similar(a)
  @test arg1(copyto!(b, a)) ≡ δ(2, 2)
  @test arg2(copyto!(b, a)) == arg2(a)
  @test arg1(permutedims(a, (2, 1))) ≡ δ(2, 2)
  @test arg2(permutedims(a, (2, 1))) == permutedims(arg2(a), (2, 1))
  b = similar(a)
  @test arg1(permutedims!(b, a, (2, 1))) ≡ δ(2, 2)
  @test arg2(permutedims!(b, a, (2, 1))) == permutedims(arg2(a), (2, 1))

  a = randn(3, 3) ⊗ δ(2, 2)
  @test size(a) == (6, 6)
  @test a + a == (2 * arg1(a)) ⊗ δ(2, 2)
  @test 2a == (2 * arg1(a)) ⊗ δ(2, 2)
  @test a * a == (arg1(a) * arg1(a)) ⊗ δ(2, 2)
  @test_broken arg2(a[(:) × (:), (:) × (:)]) ≡ δ(2, 2)
  @test_broken arg2(view(a, (:) × (:), (:) × (:))) ≡ δ(2, 2)
  @test_broken arg2(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)]) ≡ δ(2, 2)
  @test_broken arg2(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:))) ≡ δ(2, 2)
  @test_broken arg2(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡ δ(2, 2)
  @test_broken arg2(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:))) ≡ δ(2, 2)
  @test_broken arg2(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)]) ≡
    δ(2, 2)
  @test_broken arg2(
    view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:))
  ) ≡ δ(2, 2)
  @test arg2(adapt(JLArray, a)) ≡ δ(2, 2)
  @test arg1(adapt(JLArray, a)) == jl(arg1(a))
  @test arg1(adapt(JLArray, a)) isa JLArray
  @test_broken arg2(similar(a, (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡ δ(3, 3)
  @test_broken arg2(similar(typeof(a), (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡
    δ(3, 3)
  @test_broken arg2(similar(a, Float32, (cartesianrange(2 × 3), cartesianrange(2 × 3)))) ≡
    δ(Float32, (3, 3))
  @test arg2(copy(a)) ≡ δ(2, 2)
  @test arg2(copy(a)) == arg2(a)
  b = similar(a)
  @test arg2(copyto!(b, a)) ≡ δ(2, 2)
  @test arg2(copyto!(b, a)) == arg2(a)
  @test arg2(permutedims(a, (2, 1))) ≡ δ(2, 2)
  @test arg1(permutedims(a, (2, 1))) == permutedims(arg1(a), (2, 1))
  b = similar(a)
  @test arg2(permutedims!(b, a, (2, 1))) ≡ δ(2, 2)
  @test arg1(permutedims!(b, a, (2, 1))) == permutedims(arg1(a), (2, 1))

  # Views
  a = @constinferred(Eye(2) ⊗ randn(3, 3))
  b = @constinferred(view(a, (:) × (2:3), (:) × (2:3)))
  @test_broken arg1(b) ≡ Eye(2)
  @test arg2(b) ≡ view(arg2(a), 2:3, 2:3)
  @test arg2(b) == arg2(a)[2:3, 2:3]

  a = randn(3, 3) ⊗ Eye(2)
  @test size(a) == (6, 6)
  @test a + a == (2arg1(a)) ⊗ Eye(2)
  @test 2a == (2arg1(a)) ⊗ Eye(2)
  @test a * a == (arg1(a) * arg1(a)) ⊗ Eye(2)

  # Views
  a = @constinferred(randn(3, 3) ⊗ Eye(2))
  b = @constinferred(view(a, (2:3) × (:), (2:3) × (:)))
  @test arg1(b) ≡ view(arg1(a), 2:3, 2:3)
  @test arg1(b) == arg1(a)[2:3, 2:3]
  @test_broken arg2(b) ≡ Eye(2)

  # similar
  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a)
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg1(a′) ≡ arg1(a)

  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a, eltype(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg1(a′) ≡ arg1(a)

  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a, axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg1(a′) ≡ arg1(a)

  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a, eltype(a), axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg1(a′) ≡ arg1(a)

  @test_broken similar(typeof(a), axes(a))

  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a, Float32)
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{Float32,ndims(a)}
  @test_broken arg1(a′) ≡ Eye{Float32}(2)

  a = Eye(2) ⊗ randn(3, 3)
  a′ = similar(a, Float32, axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{Float32,ndims(a)}
  @test_broken arg1(a′) ≡ Eye{Float32}(2)

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a)
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg2(a′) ≡ arg2(a)

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a, eltype(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg2(a′) ≡ arg2(a)

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a, axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg2(a′) ≡ arg2(a)

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a, eltype(a), axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  @test arg2(a′) ≡ arg2(a)

  @test_broken similar(typeof(a), axes(a))

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a, Float32)
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{Float32,ndims(a)}
  # This is broken because of:
  # https://github.com/JuliaArrays/FillArrays.jl/issues/415
  @test_broken arg2(a′) ≡ Eye{Float32}(2)

  a = randn(3, 3) ⊗ Eye(2)
  a′ = similar(a, Float32, axes(a))
  @test size(a′) == (6, 6)
  @test a′ isa KroneckerArray{Float32,ndims(a)}

  a = Eye(3) ⊗ Eye(2)
  for a′ in (
    similar(a), similar(a, eltype(a)), similar(a, axes(a)), similar(a, eltype(a), axes(a))
  )
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a),ndims(a)}
  end
  @test_broken similar(typeof(a), axes(a))

  a = Eye(3) ⊗ Eye(2)
  for args in ((Float32,), (Float32, axes(a)))
    a′ = similar(a, args...)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32,ndims(a)}
  end

  # DerivableInterfaces.zero!
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    zero!(a)
    @test iszero(a)
  end
  a = Eye(3) ⊗ Eye(2)
  @test_throws ErrorException zero!(a)

  # map!(+, ...)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(+, a′, a, a)
    @test collect(a′) ≈ 2 * collect(a)
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  map!(+, a′, a, a)
  @test a′ ≈ 2a

  # map!(-, ...)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(-, a′, a, a)
    @test norm(collect(a′)) ≈ 0
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  map!(-, a′, a, a)
  @test iszero(a′)

  # map!(-, b, a)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(-, a′, a)
    @test collect(a′) ≈ -collect(a)
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  map!(-, a′, a)
  @test a′ ≈ -a

  ## # Eye ⊗ A
  ## rng = StableRNG(123)
  ## a = Eye(2) ⊗ randn(rng, 3, 3)
  ## for f in MATRIX_FUNCTIONS
  ##   @eval begin
  ##     fa = $f($a)
  ##     @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
  ##     @test arg1(fa) isa Eye
  ##   end
  ## end

  fa = inv(a)
  @test collect(fa) ≈ inv(collect(a))
  @test arg1(fa) isa Eye

  fa = pinv(a)
  @test collect(fa) ≈ pinv(collect(a))
  @test_broken arg1(fa) isa Eye

  @test det(a) ≈ det(collect(a))

  ## # A ⊗ Eye
  ## rng = StableRNG(123)
  ## a = randn(rng, 3, 3) ⊗ Eye(2)
  ## for f in setdiff(MATRIX_FUNCTIONS, [:atanh])
  ##   @eval begin
  ##     fa = $f($a)
  ##     @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
  ##     @test arg2(fa) isa Eye
  ##   end
  ## end

  fa = inv(a)
  @test collect(fa) ≈ inv(collect(a))
  @test arg2(fa) isa Eye

  fa = pinv(a)
  @test collect(fa) ≈ pinv(collect(a))
  @test_broken arg2(fa) isa Eye

  @test det(a) ≈ det(collect(a))

  # Eye ⊗ Eye
  a = Eye(2) ⊗ Eye(2)
  for f in KroneckerArrays.MATRIX_FUNCTIONS
    @eval begin
      @test $f($a) == arg1($a) ⊗ $f(arg2($a))
    end
  end

  fa = inv(a)
  @test fa == a
  @test arg1(fa) isa Eye
  @test arg2(fa) isa Eye

  fa = pinv(a)
  @test fa == a
  @test_broken arg1(fa) isa Eye
  @test_broken arg2(fa) isa Eye

  @test det(a) ≈ det(collect(a)) ≈ 1

  # permutedims
  a = Eye(2, 2) ⊗ randn(3, 3)
  @test permutedims(a, (2, 1)) == Eye(2, 2) ⊗ permutedims(arg2(a), (2, 1))

  a = randn(2, 2) ⊗ Eye(3, 3)
  @test permutedims(a, (2, 1)) == permutedims(arg1(a), (2, 1)) ⊗ Eye(3, 3)

  # permutedims!
  a = Eye(2, 2) ⊗ randn(3, 3)
  b = similar(a)
  permutedims!(b, a, (2, 1))
  @test b == Eye(2, 2) ⊗ permutedims(arg2(a), (2, 1))

  a = randn(3, 3) ⊗ Eye(2, 2)
  b = similar(a)
  permutedims!(b, a, (2, 1))
  @test b == permutedims(arg1(a), (2, 1)) ⊗ Eye(2, 2)
end

@testset "FillArrays.Zeros" begin
  a = randn(2, 2) ⊗ randn(2, 2)
  b = Zeros(2, 2) ⊗ Zeros(2, 2)
  for (x, y) in ((a, b), (b, a))
    @test x + y == a
    @test x .+ y == a
    @test map!(+, similar(a), x, y) == a
    @test (similar(a) .= x .+ y) == a
  end

  @test a - b == a
  @test a .- b == a
  @test map!(-, similar(a), a, b) == a
  @test (similar(a) .= a .- b) == a

  @test b - a == -a
  @test b .- a == -a
  @test map!(-, similar(a), b, a) == -a
  @test (similar(a) .= b .- a) == -a

  @test b + b == b
  @test b .+ b == b
  @test b - b == b
  @test b .- b == b
end
