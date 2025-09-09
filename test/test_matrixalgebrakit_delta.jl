using FillArrays: Ones
using DiagonalArrays: δ, DeltaMatrix
using KroneckerArrays: ⊗, arguments
using LinearAlgebra: Hermitian, I, diag, hermitianpart, norm
using MatrixAlgebraKit:
  eig_full,
  eig_trunc,
  eig_vals,
  eigh_full,
  eigh_trunc,
  eigh_vals,
  left_null,
  left_orth,
  left_polar,
  lq_compact,
  lq_full,
  qr_compact,
  qr_full,
  right_null,
  right_orth,
  right_polar,
  svd_compact,
  svd_full,
  svd_trunc,
  svd_vals
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred

herm(a) = parent(hermitianpart(a))

@testset "MatrixAlgebraKit + DeltaMatrix" begin
  for elt in (Float32, ComplexF32)
    a = δ(elt, 3, 3) ⊗ randn(elt, 3, 3)
    d, v = @constinferred eig_full(a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa DeltaMatrix{complex(elt)}
    @test arguments(v, 1) isa DeltaMatrix{complex(elt)}

    a = parent(hermitianpart(randn(elt, 3, 3))) ⊗ δ(elt, 3, 3)
    d, v = @constinferred eig_full(a)
    @test a * v ≈ v * d
    @test arguments(d, 2) isa DeltaMatrix{complex(elt)}
    @test arguments(v, 2) isa DeltaMatrix{complex(elt)}

    a = δ(elt, 3, 3) ⊗ δ(elt, 3, 3)
    d, v = @constinferred eig_full(a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa DeltaMatrix{complex(elt)}
    @test arguments(d, 2) isa DeltaMatrix{complex(elt)}
    @test arguments(v, 1) isa DeltaMatrix{complex(elt)}
    @test arguments(v, 2) isa DeltaMatrix{complex(elt)}
  end

  for elt in (Float32, ComplexF32)
    a = δ(elt, 3, 3) ⊗ parent(hermitianpart(randn(elt, 3, 3)))
    d, v = @constinferred eigh_full($a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa DeltaMatrix{real(elt)}
    @test arguments(v, 1) isa DeltaMatrix{elt}

    a = parent(hermitianpart(randn(elt, 3, 3))) ⊗ δ(elt, 3, 3)
    d, v = @constinferred eigh_full($a)
    @test a * v ≈ v * d
    @test arguments(d, 2) isa DeltaMatrix{real(elt)}
    @test arguments(v, 2) isa DeltaMatrix{elt}

    a = δ(elt, 3, 3) ⊗ δ(elt, 3, 3)
    d, v = @constinferred eigh_full($a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa DeltaMatrix{real(elt)}
    @test arguments(d, 2) isa DeltaMatrix{real(elt)}
    @test arguments(v, 1) isa DeltaMatrix{elt}
    @test arguments(v, 2) isa DeltaMatrix{elt}
  end

  ## TODO: Broken, need to fix truncation.
  ## for f in (eig_trunc, eigh_trunc)
  ##   a = δ(3, 3) ⊗ parent(hermitianpart(randn(3, 3)))
  ##   d, v = f(a; trunc=(; maxrank=7))
  ##   @test a * v ≈ v * d
  ##   @test arguments(d, 1) isa DeltaMatrix
  ##   @test arguments(v, 1) isa DeltaMatrix
  ##   @test size(d) == (6, 6)
  ##   @test size(v) == (9, 6)

  ##   a = parent(hermitianpart(randn(3, 3))) ⊗ δ(3, 3)
  ##   d, v = f(a; trunc=(; maxrank=7))
  ##   @test a * v ≈ v * d
  ##   @test arguments(d, 2) isa DeltaMatrix
  ##   @test arguments(v, 2) isa DeltaMatrix
  ##   @test size(d) == (6, 6)
  ##   @test size(v) == (9, 6)

  ##   a = δ(3, 3) ⊗ δ(3, 3)
  ##   @test_throws ArgumentError f(a)
  ## end

  for f in (eig_vals, eigh_vals)
    a = δ(3, 3) ⊗ parent(hermitianpart(randn(3, 3)))
    d = @constinferred f(a)
    d′ = f(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 1) isa Ones
    @test arguments(d, 2) ≈ f(arguments(a, 2))

    a = parent(hermitianpart(randn(3, 3))) ⊗ δ(3, 3)
    d = @constinferred f(a)
    d′ = f(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 2) isa Ones
    @test arguments(d, 1) ≈ f(arguments(a, 1))

    a = δ(3, 3) ⊗ δ(3, 3)
    d = @constinferred f(a)
    @test d == Ones(3) ⊗ Ones(3)
    @test arguments(d, 1) isa Ones
    @test arguments(d, 2) isa Ones
  end

  for f in (
    left_orth, right_orth, left_polar, right_polar, qr_compact, lq_compact, qr_full, lq_full
  )
    a = δ(3, 3) ⊗ randn(3, 3)
    if VERSION ≥ v"1.11-"
      x, y = @constinferred f($a)
    else
      # Type inference fails in Julia 1.10.
      x, y = f(a)
    end
    @test x * y ≈ a
    @test arguments(x, 1) isa DeltaMatrix
    @test arguments(y, 1) isa DeltaMatrix

    a = randn(3, 3) ⊗ δ(3, 3)
    x, y = @constinferred f($a)
    @test x * y ≈ a
    @test arguments(x, 2) isa DeltaMatrix
    @test arguments(y, 2) isa DeltaMatrix

    a = δ(3, 3) ⊗ δ(3, 3)
    x, y = @constinferred f($a)
    @test x * y ≈ a
    @test arguments(x, 1) isa DeltaMatrix
    @test arguments(y, 1) isa DeltaMatrix
    @test arguments(x, 2) isa DeltaMatrix
    @test arguments(y, 2) isa DeltaMatrix
  end

  for f in (svd_compact, svd_full)
    for elt in (Float32, ComplexF32)
      a = δ(elt, 3, 3) ⊗ randn(elt, 3, 3)
      u, s, v = @constinferred f($a)
      @test u * s * v ≈ a
      @test eltype(u) === elt
      @test eltype(s) === real(elt)
      @test eltype(v) === elt
      @test arguments(u, 1) isa DeltaMatrix{elt}
      @test arguments(s, 1) isa DeltaMatrix{real(elt)}
      @test arguments(v, 1) isa DeltaMatrix{elt}

      a = randn(elt, 3, 3) ⊗ δ(elt, 3, 3)
      u, s, v = @constinferred f($a)
      @test u * s * v ≈ a
      @test eltype(u) === elt
      @test eltype(s) === real(elt)
      @test eltype(v) === elt
      @test arguments(u, 2) isa DeltaMatrix{elt}
      @test arguments(s, 2) isa DeltaMatrix{real(elt)}
      @test arguments(v, 2) isa DeltaMatrix{elt}

      a = δ(elt, 3, 3) ⊗ δ(elt, 3, 3)
      u, s, v = @constinferred f($a)
      @test u * s * v ≈ a
      @test eltype(u) === elt
      @test eltype(s) === real(elt)
      @test eltype(v) === elt
      @test arguments(u, 1) isa DeltaMatrix{elt}
      @test arguments(s, 1) isa DeltaMatrix{real(elt)}
      @test arguments(v, 1) isa DeltaMatrix{elt}
      @test arguments(u, 2) isa DeltaMatrix{elt}
      @test arguments(s, 2) isa DeltaMatrix{real(elt)}
      @test arguments(v, 2) isa DeltaMatrix{elt}
    end
  end

  ## TODO: Need to implement truncation.
  ## # svd_trunc
  ## for elt in (Float32, ComplexF32)
  ##   a = δ(elt, 3, 3) ⊗ randn(elt, 3, 3)
  ##   # TODO: Type inference is broken for `svd_trunc`,
  ##   # look into fixing it.
  ##   # u, s, v = @constinferred svd_trunc(a; trunc=(; maxrank=7))
  ##   u, s, v = svd_trunc(a; trunc=(; maxrank=7))
  ##   @test eltype(u) === elt
  ##   @test eltype(s) === real(elt)
  ##   @test eltype(v) === elt
  ##   u′, s′, v′ = svd_trunc(Matrix(a); trunc=(; maxrank=6))
  ##   @test Matrix(u * s * v) ≈ u′ * s′ * v′
  ##   @test arguments(u, 1) isa DeltaMatrix{elt}
  ##   @test arguments(s, 1) isa DeltaMatrix{real(elt)}
  ##   @test arguments(v, 1) isa DeltaMatrix{elt}
  ##   @test size(u) == (9, 6)
  ##   @test size(s) == (6, 6)
  ##   @test size(v) == (6, 9)
  ## end

  ## TODO: Need to implement truncation.
  ## for elt in (Float32, ComplexF32)
  ##   a = randn(elt, 3, 3) ⊗ δ(elt, 3, 3)
  ##   # TODO: Type inference is broken for `svd_trunc`,
  ##   # look into fixing it.
  ##   # u, s, v = @constinferred svd_trunc(a; trunc=(; maxrank=7))
  ##   u, s, v = svd_trunc(a; trunc=(; maxrank=7))
  ##   @test eltype(u) === elt
  ##   @test eltype(s) === real(elt)
  ##   @test eltype(v) === elt
  ##   u′, s′, v′ = svd_trunc(Matrix(a); trunc=(; maxrank=6))
  ##   @test Matrix(u * s * v) ≈ u′ * s′ * v′
  ##   @test arguments(u, 2) isa DeltaMatrix{elt}
  ##   @test arguments(s, 2) isa DeltaMatrix{real(elt)}
  ##   @test arguments(v, 2) isa DeltaMatrix{elt}
  ##   @test size(u) == (9, 6)
  ##   @test size(s) == (6, 6)
  ##   @test size(v) == (6, 9)
  ## end

  a = δ(3, 3) ⊗ δ(3, 3)
  @test_broken svd_trunc(a)

  # svd_vals
  for elt in (Float32, ComplexF32)
    a = δ(elt, 3, 3) ⊗ randn(elt, 3, 3)
    d = @constinferred svd_vals(a)
    d′ = svd_vals(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 1) isa Ones{real(elt)}
    @test arguments(d, 2) ≈ svd_vals(arguments(a, 2))
  end

  for elt in (Float32, ComplexF32)
    a = randn(elt, 3, 3) ⊗ δ(elt, 3, 3)
    d = @constinferred svd_vals(a)
    d′ = svd_vals(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 2) isa Ones{real(elt)}
    @test arguments(d, 1) ≈ svd_vals(arguments(a, 1))
  end

  for elt in (Float32, ComplexF32)
    a = δ(elt, 3, 3) ⊗ δ(elt, 3, 3)
    d = @constinferred svd_vals(a)
    @test d ≡ Ones{real(elt)}(3) ⊗ Ones{real(elt)}(3)
    @test arguments(d, 1) isa Ones{real(elt)}
    @test arguments(d, 2) isa Ones{real(elt)}
  end

  # left_null
  a = δ(3, 3) ⊗ randn(3, 3)
  @test_broken left_null(a)
  ## n = @constinferred left_null(a)
  ## @test norm(n' * a) ≈ 0
  ## @test arguments(n, 1) isa DeltaMatrix

  a = randn(3, 3) ⊗ δ(3, 3)
  @test_broken left_null(a)
  ## n = @constinferred left_null(a)
  ## @test norm(n' * a) ≈ 0
  ## @test arguments(n, 2) isa DeltaMatrix

  a = δ(3, 3) ⊗ δ(3, 3)
  @test_broken left_null(a)

  # right_null
  a = δ(3, 3) ⊗ randn(3, 3)
  @test_broken right_null(a)
  ## n = @constinferred right_null(a)
  ## @test norm(a * n') ≈ 0
  ## @test arguments(n, 1) isa DeltaMatrix

  a = randn(3, 3) ⊗ δ(3, 3)
  @test_broken right_null(a)
  ## n = @constinferred right_null(a)
  ## @test norm(a * n') ≈ 0
  ## @test arguments(n, 2) isa DeltaMatrix

  a = δ(3, 3) ⊗ δ(3, 3)
  @test_broken right_null(a)
end
