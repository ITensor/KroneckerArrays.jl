using KroneckerArrays: ⊗
using LinearAlgebra: Hermitian, diag
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
  x = randn(2, 2)
  y = randn(3, 3)
  a = x ⊗ y
  ah = Hermitian(x) ⊗ Hermitian(y)

  d, v = eig_full(a)
  @test a * v ≈ v * d

  @test_throws MethodError eig_trunc(a)

  d = eig_vals(a)
  @test d ≈ diag(eig_full(a)[1])

  d, v = eigh_full(ah)
  @test ah * v ≈ v * d

  @test_throws MethodError eigh_trunc(ah)

  d = eigh_vals(ah)
  @test d ≈ diag(eigh_full(ah)[1])
end
