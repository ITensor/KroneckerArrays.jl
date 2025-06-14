using DerivableInterfaces: zero!
using FillArrays: Eye
using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗
using LinearAlgebra: det, norm, pinv
using StableRNGs: StableRNG
using Test: @test, @testset

@testset "FillArrays.Eye" begin
  MATRIX_FUNCTIONS = KroneckerArrays.MATRIX_FUNCTIONS
  if VERSION < v"1.11-"
    # `cbrt(::AbstractMatrix{<:Real})` was implemented in Julia 1.11.
    MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, [:cbrt])
  end

  a = Eye(2) ⊗ randn(3, 3)
  @test size(a) == (6, 6)
  @test a + a == Eye(2) ⊗ (2a.b)
  @test 2a == Eye(2) ⊗ (2a.b)
  @test a * a == Eye(2) ⊗ (a.b * a.b)

  a = randn(3, 3) ⊗ Eye(2)
  @test size(a) == (6, 6)
  @test a + a == (2a.a) ⊗ Eye(2)
  @test 2a == (2a.a) ⊗ Eye(2)
  @test a * a == (a.a * a.a) ⊗ Eye(2)

  # similar
  a = Eye(2) ⊗ randn(3, 3)
  for a′ in (
    similar(a),
    similar(a, eltype(a)),
    similar(a, axes(a)),
    similar(a, eltype(a), axes(a)),
    similar(typeof(a), axes(a)),
  )
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a),ndims(a),typeof(a.a),typeof(a.b)}
    @test a′.a === a.a
  end

  a = Eye(2) ⊗ randn(3, 3)
  for args in ((Float32,), (Float32, axes(a)))
    a′ = similar(a, args...)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32,ndims(a)}
    @test a′.a === Eye{Float32}(2)
  end

  a = randn(3, 3) ⊗ Eye(2)
  for a′ in (
    similar(a),
    similar(a, eltype(a)),
    similar(a, axes(a)),
    similar(a, eltype(a), axes(a)),
    similar(typeof(a), axes(a)),
  )
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a),ndims(a),typeof(a.a),typeof(a.b)}
    @test a′.b === a.b
  end

  a = randn(3, 3) ⊗ Eye(2)
  for args in ((Float32,), (Float32, axes(a)))
    a′ = similar(a, args...)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32,ndims(a)}
    @test a′.b === Eye{Float32}(2)
  end

  a = Eye(3) ⊗ Eye(2)
  for a′ in (
    similar(a),
    similar(a, eltype(a)),
    similar(a, axes(a)),
    similar(a, eltype(a), axes(a)),
    similar(typeof(a), axes(a)),
  )
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a),ndims(a),typeof(a.a),typeof(a.b)}
    @test a′.a === a.a
    @test a′.b === a.b
  end

  a = Eye(3) ⊗ Eye(2)
  for args in ((Float32,), (Float32, axes(a)))
    a′ = similar(a, args...)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32,ndims(a)}
    @test a′.a === Eye{Float32}(3)
    @test a′.b === Eye{Float32}(2)
  end

  # DerivableInterfaces.zero!
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    zero!(a)
    @test iszero(a)
  end
  a = Eye(3) ⊗ Eye(2)
  @test_throws ArgumentError zero!(a)

  # map!(+, ...)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(+, a′, a, a)
    @test collect(a′) ≈ 2 * collect(a)
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  @test_throws ErrorException map!(+, a′, a, a)

  # map!(-, ...)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(-, a′, a, a)
    @test norm(collect(a′)) ≈ 0
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  @test_throws ErrorException map!(-, a′, a, a)

  # map!(-, b, a)
  for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
    a′ = similar(a)
    map!(-, a′, a)
    @test collect(a′) ≈ -collect(a)
  end
  a = Eye(3) ⊗ Eye(2)
  a′ = similar(a)
  @test_throws ErrorException map!(-, a′, a)

  # Eye ⊗ A
  rng = StableRNG(123)
  a = Eye(2) ⊗ randn(rng, 3, 3)
  for f in MATRIX_FUNCTIONS
    @eval begin
      fa = $f($a)
      @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
      @test fa.a isa Eye
    end
  end

  fa = inv(a)
  @test collect(fa) ≈ inv(collect(a))
  @test fa.a isa Eye

  fa = pinv(a)
  @test collect(fa) ≈ pinv(collect(a))
  @test fa.a isa Eye

  @test det(a) ≈ det(collect(a))

  # A ⊗ Eye
  rng = StableRNG(123)
  a = randn(rng, 3, 3) ⊗ Eye(2)
  for f in setdiff(MATRIX_FUNCTIONS, [:atanh])
    @eval begin
      fa = $f($a)
      @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
      @test fa.b isa Eye
    end
  end

  fa = inv(a)
  @test collect(fa) ≈ inv(collect(a))
  @test fa.b isa Eye

  fa = pinv(a)
  @test collect(fa) ≈ pinv(collect(a))
  @test fa.b isa Eye

  @test det(a) ≈ det(collect(a))

  # Eye ⊗ Eye
  a = Eye(2) ⊗ Eye(2)
  for f in KroneckerArrays.MATRIX_FUNCTIONS
    @eval begin
      @test_throws ArgumentError $f($a)
    end
  end

  fa = inv(a)
  @test fa == a
  @test fa.a isa Eye
  @test fa.b isa Eye

  fa = pinv(a)
  @test fa == a
  @test fa.a isa Eye
  @test fa.b isa Eye

  @test det(a) ≈ det(collect(a)) ≈ 1
end
