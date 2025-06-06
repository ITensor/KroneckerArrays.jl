using FillArrays: Eye
using KroneckerArrays: KroneckerArrays, ⊗, ×, diagonal, kron_nd
using LinearAlgebra: Diagonal, I, det, eigen, eigvals, lq, pinv, qr, svd, svdvals, tr
using Test: @test, @test_broken, @test_throws, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)
@testset "KroneckerArrays (eltype=$elt)" for elt in elts
  p = [1, 2] × [3, 4, 5]
  @test length(p) == 6
  @test collect(p) == [1 × 3, 2 × 3, 1 × 4, 2 × 4, 1 × 5, 2 × 5]

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  b = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c = a.a ⊗ b.b
  @test similar(typeof(a), (2, 3)) isa Matrix{elt}
  @test size(similar(typeof(a), (2, 3))) == (2, 3)
  @test a[1 × 1, 1 × 1] == a.a[1, 1] * a.b[1, 1]
  @test a[1 × 3, 2 × 1] == a.a[1, 2] * a.b[3, 1]
  @test a[1 × (2:3), 2 × 1] == a.a[1, 2] * a.b[2:3, 1]
  @test a[1 × :, (:) × 1] == a.a[1, :] ⊗ a.b[:, 1]
  @test a[(1:2) × (2:3), (1:2) × (2:3)] == a.a[1:2, 1:2] ⊗ a.b[2:3, 2:3]
  v = randn(elt, 2) ⊗ randn(elt, 3)
  @test v[1 × 1] == v.a[1] * v.b[1]
  @test v[1 × 3] == v.a[1] * v.b[3]
  @test v[(1:2) × 3] == v.a[1:2] * v.b[3]
  @test v[(1:2) × (2:3)] == v.a[1:2] ⊗ v.b[2:3]
  @test eltype(a) === elt
  @test collect(a) == kron(collect(a.a), collect(a.b))
  @test size(a) == (6, 6)
  @test collect(a * b) ≈ collect(a) * collect(b)
  @test collect(-a) == -collect(a)
  @test collect(3 * a) ≈ 3 * collect(a)
  @test collect(a * 3) ≈ collect(a) * 3
  @test a + a == 2a
  @test iszero(a - a)
  @test collect(a + c) ≈ collect(a) + collect(c)
  @test collect(b + c) ≈ collect(b) + collect(c)
  for f in (transpose, adjoint, inv, pinv)
    @test collect(f(a)) ≈ f(collect(a))
  end
  @test tr(a) ≈ tr(collect(a))

  a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
  @test collect(a) ≈ kron_nd(a.a, a.b)
  @test a[1 × 1, 1 × 1, 1 × 1] == a.a[1, 1, 1] * a.b[1, 1, 1]
  @test a[1 × 3, 2 × 1, 2 × 2] == a.a[1, 2, 2] * a.b[3, 1, 2]
  @test collect(a + a) ≈ 2 * collect(a)
  @test collect(2a) ≈ 2 * collect(a)

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  b = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c = a.a ⊗ b.b
  U, S, V = svd(a)
  @test collect(U * diagonal(S) * V') ≈ collect(a)
  @test svdvals(a) ≈ S
  @test sort(collect(S); rev=true) ≈ svdvals(collect(a))
  @test collect(U'U) ≈ I
  @test collect(V * V') ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  D, V = eigen(a)
  @test collect(a * V) ≈ collect(V * diagonal(D))
  @test eigvals(a) ≈ D

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  Q, R = qr(a)
  @test collect(Q * R) ≈ collect(a)
  @test collect(Q'Q) ≈ I

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  @test det(a) ≈ det(collect(a))

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  for f in KroneckerArrays.MATRIX_FUNCTIONS
    @eval begin
      @test_throws ArgumentError $f($a)
    end
  end
end

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

  # Eye ⊗ A
  a = Eye(2) ⊗ randn(3, 3)
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
  a = randn(3, 3) ⊗ Eye(2)
  for f in MATRIX_FUNCTIONS
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
