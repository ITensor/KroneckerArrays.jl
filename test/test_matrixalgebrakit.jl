using KroneckerArrays: ⊗
using LinearAlgebra: Hermitian, diag, norm
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

  a = Hermitian(randn(elt, 2, 2)) ⊗ Hermitian(randn(elt, 3, 3))
  d, v = eigh_full(a)
  @test a * v ≈ v * d

  a = Hermitian(randn(elt, 2, 2)) ⊗ Hermitian(randn(elt, 3, 3))
  @test_throws MethodError eigh_trunc(a)

  a = Hermitian(randn(elt, 2, 2)) ⊗ Hermitian(randn(elt, 3, 3))
  d = eigh_vals(a)
  @test d ≈ diag(eigh_full(a)[1])

  a = randn(elt, 3, 2) ⊗ randn(elt, 4, 3)
  n = left_null(a)
  @test norm(n' * a) ≈ 0 atol = √eps(real(elt))

  a = randn(elt, 2, 3) ⊗ randn(elt, 3, 4)
  n = right_null(a)
  @test norm(a * n') ≈ 0 atol = √eps(real(elt))
end
