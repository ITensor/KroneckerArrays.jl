using FillArrays: Eye, Ones
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
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

herm(a) = parent(hermitianpart(a))

@testset "MatrixAlgebraKit" begin
  elt = Float32

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  d, v = eig_full(a)
  @test a * v ≈ v * d

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  @test_throws MethodError eig_trunc(a)

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  d = eig_vals(a)
  @test d ≈ diag(eig_full(a)[1])

  a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
  d, v = eigh_full(a)
  @test a * v ≈ v * d

  a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
  @test_throws MethodError eigh_trunc(a)

  a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
  d = eigh_vals(a)
  @test d ≈ diag(eigh_full(a)[1])

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, c = qr_compact(a)
  @test u * c ≈ a
  @test collect(u'u) ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, c = qr_full(a)
  @test u * c ≈ a
  @test collect(u'u) ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c, u = lq_compact(a)
  @test c * u ≈ a
  @test collect(u * u') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c, u = lq_full(a)
  @test c * u ≈ a
  @test collect(u * u') ≈ I

  a = randn(elt, 3, 2) ⊗ randn(elt, 4, 3)
  n = left_null(a)
  @test norm(n' * a) ≈ 0 atol = √eps(real(elt))

  a = randn(elt, 2, 3) ⊗ randn(elt, 3, 4)
  n = right_null(a)
  @test norm(a * n') ≈ 0 atol = √eps(real(elt))

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, c = left_orth(a)
  @test u * c ≈ a
  @test collect(u'u) ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c, u = right_orth(a)
  @test c * u ≈ a
  @test collect(u * u') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, c = left_polar(a)
  @test u * c ≈ a
  @test collect(u'u) ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c, u = right_polar(a)
  @test c * u ≈ a
  @test collect(u * u') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, s, v = svd_compact(a)
  @test u * s * v ≈ a
  @test collect(u'u) ≈ I
  @test collect(v * v') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  u, s, v = svd_full(a)
  @test u * s * v ≈ a
  @test collect(u'u) ≈ I
  @test collect(v * v') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  @test_throws MethodError svd_trunc(a)

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  s = svd_vals(a)
  @test s ≈ diag(svd_compact(a)[2])
end

@testset "MatrixAlgebraKit + Eye" begin
  for f in (eig_full, eigh_full)
    a = Eye(3) ⊗ parent(hermitianpart(randn(3, 3)))
    d, v = @constinferred f(a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa Eye
    @test arguments(v, 1) isa Eye

    a = parent(hermitianpart(randn(3, 3))) ⊗ Eye(3)
    d, v = @constinferred f(a)
    @test a * v ≈ v * d
    @test arguments(d, 2) isa Eye
    @test arguments(v, 2) isa Eye

    a = Eye(3) ⊗ Eye(3)
    d, v = @constinferred f(a)
    @test a * v ≈ v * d
    @test arguments(d, 1) isa Eye
    @test arguments(d, 2) isa Eye
    @test arguments(v, 1) isa Eye
    @test arguments(v, 2) isa Eye
  end

  for f in (eig_trunc, eigh_trunc)
    a = Eye(3) ⊗ parent(hermitianpart(randn(3, 3)))
    d, v = f(a; trunc=(; maxrank=7))
    @test a * v ≈ v * d
    @test arguments(d, 1) isa Eye
    @test arguments(v, 1) isa Eye
    @test size(d) == (6, 6)
    @test size(v) == (9, 6)

    a = parent(hermitianpart(randn(3, 3))) ⊗ Eye(3)
    d, v = f(a; trunc=(; maxrank=7))
    @test a * v ≈ v * d
    @test arguments(d, 2) isa Eye
    @test arguments(v, 2) isa Eye
    @test size(d) == (6, 6)
    @test size(v) == (9, 6)

    a = Eye(3) ⊗ Eye(3)
    @test_throws ArgumentError f(a)
  end

  for f in (eig_vals, eigh_vals)
    a = Eye(3) ⊗ parent(hermitianpart(randn(3, 3)))
    d = @constinferred f(a)
    d′ = f(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 1) isa Ones
    @test arguments(d, 2) ≈ f(arguments(a, 2))

    a = parent(hermitianpart(randn(3, 3))) ⊗ Eye(3)
    d = @constinferred f(a)
    d′ = f(Matrix(a))
    @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
    @test arguments(d, 2) isa Ones
    @test arguments(d, 1) ≈ f(arguments(a, 1))

    a = Eye(3) ⊗ Eye(3)
    d = @constinferred f(a)
    @test d == Ones(3) ⊗ Ones(3)
    @test arguments(d, 1) isa Ones
    @test arguments(d, 2) isa Ones
  end

  for f in (
    left_orth, left_polar, lq_compact, lq_full, qr_compact, qr_full, right_orth, right_polar
  )
    a = Eye(3) ⊗ randn(3, 3)
    x, y = @constinferred f(a)
    @test x * y ≈ a
    @test arguments(x, 1) isa Eye
    @test arguments(y, 1) isa Eye

    a = randn(3, 3) ⊗ Eye(3)
    x, y = @constinferred f(a)
    @test x * y ≈ a
    @test arguments(x, 2) isa Eye
    @test arguments(y, 2) isa Eye

    a = Eye(3) ⊗ Eye(3)
    x, y = f(a)
    @test x * y ≈ a
    @test arguments(x, 1) isa Eye
    @test arguments(y, 1) isa Eye
    @test arguments(x, 2) isa Eye
    @test arguments(y, 2) isa Eye
  end

  for f in (svd_compact, svd_full)
    a = Eye(3) ⊗ randn(3, 3)
    u, s, v = @constinferred f(a)
    @test u * s * v ≈ a
    @test arguments(u, 1) isa Eye
    @test arguments(s, 1) isa Eye
    @test arguments(v, 1) isa Eye

    a = randn(3, 3) ⊗ Eye(3)
    u, s, v = @constinferred f(a)
    @test u * s * v ≈ a
    @test arguments(u, 2) isa Eye
    @test arguments(s, 2) isa Eye
    @test arguments(v, 2) isa Eye

    a = Eye(3) ⊗ Eye(3)
    u, s, v = @constinferred f(a)
    @test u * s * v ≈ a
    @test arguments(u, 1) isa Eye
    @test arguments(s, 1) isa Eye
    @test arguments(v, 1) isa Eye
    @test arguments(u, 2) isa Eye
    @test arguments(s, 2) isa Eye
    @test arguments(v, 2) isa Eye
  end

  # svd_trunc
  a = Eye(3) ⊗ randn(3, 3)
  u, s, v = svd_trunc(a; trunc=(; maxrank=7))
  u′, s′, v′ = svd_trunc(Matrix(a); trunc=(; maxrank=6))
  @test Matrix(u * s * v) ≈ u′ * s′ * v′
  @test arguments(u, 1) isa Eye
  @test arguments(s, 1) isa Eye
  @test arguments(v, 1) isa Eye
  @test size(u) == (9, 6)
  @test size(s) == (6, 6)
  @test size(v) == (6, 9)

  a = randn(3, 3) ⊗ Eye(3)
  u, s, v = svd_trunc(a; trunc=(; maxrank=7))
  u′, s′, v′ = svd_trunc(Matrix(a); trunc=(; maxrank=6))
  @test Matrix(u * s * v) ≈ u′ * s′ * v′
  @test arguments(u, 2) isa Eye
  @test arguments(s, 2) isa Eye
  @test arguments(v, 2) isa Eye
  @test size(u) == (9, 6)
  @test size(s) == (6, 6)
  @test size(v) == (6, 9)

  a = Eye(3) ⊗ Eye(3)
  @test_throws ArgumentError svd_trunc(a)

  # svd_vals
  a = Eye(3) ⊗ randn(3, 3)
  d = @constinferred svd_vals(a)
  d′ = svd_vals(Matrix(a))
  @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
  @test arguments(d, 1) isa Ones
  @test arguments(d, 2) ≈ svd_vals(arguments(a, 2))

  a = randn(3, 3) ⊗ Eye(3)
  d = @constinferred svd_vals(a)
  d′ = svd_vals(Matrix(a))
  @test sort(Vector(d); by=abs) ≈ sort(d′; by=abs)
  @test arguments(d, 2) isa Ones
  @test arguments(d, 1) ≈ svd_vals(arguments(a, 1))

  a = Eye(3) ⊗ Eye(3)
  d = @constinferred svd_vals(a)
  @test d == Ones(3) ⊗ Ones(3)
  @test arguments(d, 1) isa Ones
  @test arguments(d, 2) isa Ones

  # left_null
  a = Eye(3) ⊗ randn(3, 3)
  n = @constinferred left_null(a)
  @test norm(n' * a) ≈ 0
  @test arguments(n, 1) isa Eye

  a = randn(3, 3) ⊗ Eye(3)
  n = @constinferred left_null(a)
  @test norm(n' * a) ≈ 0
  @test arguments(n, 2) isa Eye

  a = Eye(3) ⊗ Eye(3)
  @test_throws MethodError left_null(a)

  # right_null
  a = Eye(3) ⊗ randn(3, 3)
  n = @constinferred right_null(a)
  @test norm(a * n') ≈ 0
  @test arguments(n, 1) isa Eye

  a = randn(3, 3) ⊗ Eye(3)
  n = @constinferred right_null(a)
  @test norm(a * n') ≈ 0
  @test arguments(n, 2) isa Eye

  a = Eye(3) ⊗ Eye(3)
  @test_throws MethodError right_null(a)
end
