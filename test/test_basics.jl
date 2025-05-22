using KroneckerArrays: KroneckerArrays, ⊗, ×, diagonal
using LinearAlgebra: Diagonal, I, eigen, eigvals, lq, qr, svd, svdvals, tr
using Test: @test, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)
@testset "KroneckerArrays (eltype=$elt)" for elt in elts
  p = [1, 2] × [3, 4, 5]
  @test length(p) == 6
  @test collect(p) == [1 × 3, 2 × 3, 1 × 4, 2 × 4, 1 × 5, 2 × 5]

  a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  b = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
  c = a.a ⊗ b.b
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
  for f in (transpose, adjoint, inv)
    @test collect(f(a)) ≈ f(collect(a))
  end
  @test tr(a) ≈ tr(collect(a))

  U, S, V = svd(a)
  @test collect(U * diagonal(S) * V') ≈ collect(a)
  @test svdvals(a) ≈ S
  @test sort(collect(S); rev=true) ≈ svdvals(collect(a))
  @test collect(U'U) ≈ I
  @test collect(V * V') ≈ I

  D, V = eigen(a)
  @test collect(a * V) ≈ collect(V * diagonal(D))
  @test eigvals(a) ≈ D

  Q, R = qr(a)
  @test collect(Q * R) ≈ collect(a)
  @test collect(Q'Q) ≈ I
end
