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
    @test_throws ArgumentError eig_trunc(a)

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    d = eig_vals(a)
    @test d ≈ diag(eig_full(a)[1])

    a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
    d, v = eigh_full(a)
    @test a * v ≈ v * d
    @test eltype(d) === real(elt)
    @test eltype(v) === elt

    a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
    @test_throws ArgumentError eigh_trunc(a)

    a = herm(randn(elt, 2, 2)) ⊗ herm(randn(elt, 3, 3))
    d = eigh_vals(a)
    @test d ≈ diag(eigh_full(a)[1])
    @test eltype(d) === real(elt)

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
    @test eltype(u) === elt
    @test eltype(s) === real(elt)
    @test eltype(v) === elt
    @test collect(u'u) ≈ I
    @test collect(v * v') ≈ I

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    u, s, v = svd_full(a)
    @test u * s * v ≈ a
    @test eltype(u) === elt
    @test eltype(s) === real(elt)
    @test eltype(v) === elt
    @test collect(u'u) ≈ I
    @test collect(v * v') ≈ I

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    @test_throws ArgumentError svd_trunc(a)

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    s = svd_vals(a)
    @test s ≈ diag(svd_compact(a)[2])
end
