using Adapt: adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted
using DiagonalArrays: diagonal
using FunctionImplementations: zero!
using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using KroneckerArrays: KroneckerArrays, KroneckerArray, KroneckerStyle,
    CartesianProductUnitRange, CartesianProductVector, ⊗, ×, kroneckerfactors, kroneckerfactortypes,
    cartesianproduct, cartesianrange, kron_nd, unproduct
using LinearAlgebra: Diagonal, I, det, eigen, eigvals, lq, norm, pinv, qr, svd, svdvals, tr
using StableRNGs: StableRNG
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred

elts = (Float32, Float64, ComplexF32, ComplexF64)
@testset "KroneckerArrays (eltype=$elt)" for elt in elts
    p = [1, 2] × [3, 4, 5]
    @test length(p) == 6
    @test collect(p) == [1 × 3, 1 × 4, 1 × 5, 2 × 3, 2 × 4, 2 × 5]

    r = @constinferred cartesianrange(2, 3)
    @test r ===
        @constinferred(cartesianrange(2 × 3)) ===
        @constinferred(cartesianrange(Base.OneTo(2), Base.OneTo(3))) ===
        @constinferred(Base.OneTo(2) × Base.OneTo(3))
    @test unproduct(r) === Base.OneTo(6)
    @test length(r) == 6
    @test first(r) == 1
    @test last(r) == 6
    @test r[1 × 1] == 1
    @test r[1 × 2] == 2
    @test r[1 × 3] == 3
    @test r[2 × 1] == 4
    @test r[2 × 2] == 5
    @test r[2 × 3] == 6

    @test sprint(show, cartesianrange(2, 3)) == "(Base.OneTo(2) × Base.OneTo(3))"
    @test sprint(show, cartesianrange(2, 3, 2:7)) == "cartesianrange(Base.OneTo(2), Base.OneTo(3), 2:7)"

    # CartesianProductUnitRange axes
    r = cartesianrange(2:3, 3:4, 2:5)
    @test axes(r, 1) ≡ cartesianrange(2, 2)

    # CartesianProductUnitRange getindex
    r1 = cartesianrange(2:4, 3:5, 2:10)
    r2 = cartesianrange(2:3, 2:3, 2:5)
    @test r1[r2] ≡ cartesianrange(3:4, 4:5, 3:6)

    @test axes(r, 1) ≡ cartesianrange(2, 2)

    # CartesianProductVector axes
    r = cartesianproduct([2, 4], [3, 5], [3, 5, 7, 9])
    @test axes(r) ≡ (cartesianrange(2, 2),)

    r = @constinferred(cartesianrange(2 × 3, 2:7))
    @test r === cartesianrange(Base.OneTo(2), Base.OneTo(3), 2:7)
    @test axes(r, 1) === Base.OneTo(2) × Base.OneTo(3)
    @test unproduct(r) === 2:7
    @test length(r) == 6
    @test first(r) == 2
    @test last(r) == 7
    @test r[1 × 1] == 2
    @test r[1 × 2] == 3
    @test r[1 × 3] == 4
    @test r[2 × 1] == 5
    @test r[2 × 2] == 6
    @test r[2 × 3] == 7

    # Test high-dimensional materialization.
    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 2, 2, 2)
    x = Array(a)
    y = similar(x)
    for I in eachindex(a)
        y[I] = @allowscalar x[I]
    end
    @test x == y

    rng = StableRNG(123)
    a = @constinferred(randn(rng, elt, 2, 2) ⊗ randn(rng, elt, 3, 3))
    b = @constinferred(randn(rng, elt, 2, 2) ⊗ randn(rng, elt, 3, 3))
    c = @constinferred(kroneckerfactors(a, 1) ⊗ kroneckerfactors(b, 2))
    @test a isa KroneckerArray{elt, 2, kroneckerfactortypes(a)...}
    @test similar(typeof(a), (2, 3)) isa Matrix{elt}
    @test size(similar(typeof(a), (2, 3))) == (2, 3)
    @test isreal(a) == (elt <: Real)
    aa, ab = kroneckerfactors(a)
    for i in 1:2, j in 1:3, k in 1:2, l in 1:3
        @test a[i × j, k × l] == aa[i, k] * ab[j, l]
    end
    @test a[1 × (2:3), 2 × 1] == aa[1, 2] * ab[2:3, 1]
    @test a[1 × :, (:) × 1] == aa[1, :] ⊗ ab[:, 1]
    @test a[(1:2) × (2:3), (1:2) × (2:3)] == aa[1:2, 1:2] ⊗ ab[2:3, 2:3]
    v = randn(elt, 2) ⊗ randn(elt, 3)
    va, vb = kroneckerfactors(v)
    @test v[1 × 1] == va[1] * vb[1]
    @test v[1 × 3] == va[1] * vb[3]
    @test v[(1:2) × 3] == va[1:2] * vb[3]
    @test v[(1:2) × (2:3)] == va[1:2] ⊗ vb[2:3]
    @test eltype(a) === elt
    @test collect(a) == kron(collect(aa), collect(ab))
    @test size(a) == (6, 6)
    @test collect(a * b) ≈ collect(a) * collect(b)
    @test collect(-a) == -collect(a)
    @test collect(3 * a) ≈ 3 * collect(a)
    @test collect(a * 3) ≈ collect(a) * 3
    @test collect(a / 3) ≈ collect(a) / 3
    @test a + a == 2a
    @test iszero(a - a)
    @test collect(a + c) ≈ collect(a) + collect(c)
    @test collect(b + c) ≈ collect(b) + collect(c)
    for f in (transpose, adjoint, inv, pinv)
        @test collect(f(a)) ≈ f(collect(a))
    end
    @test tr(a) ≈ tr(collect(a))
    @test norm(a) ≈ norm(collect(a))

    # Views
    a = @constinferred(randn(elt, 2, 2) ⊗ randn(elt, 3, 3))
    b = @constinferred(view(a, (1:2) × (2:3), (1:2) × (2:3)))
    @test kroneckerfactors(b, 1) === view(kroneckerfactors(a, 1), 1:2, 1:2)
    @test kroneckerfactors(b, 1) == kroneckerfactors(a, 1)[1:2, 1:2]
    @test kroneckerfactors(b, 2) === view(kroneckerfactors(a, 2), 2:3, 2:3)
    @test kroneckerfactors(b, 2) == kroneckerfactors(a, 2)[2:3, 2:3]

    # Broadcasting
    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    style = KroneckerStyle(BroadcastStyle.(kroneckerfactortypes(a))...)
    @test BroadcastStyle(typeof(a)) === style
    @test_throws "not supported" sin.(a)
    a′ = similar(a)
    @test_throws "not supported" a′ .= sin.(a)
    a′ = similar(a)
    a′ .= 2 .* a
    @test collect(a′) ≈ 2 * collect(a)
    bc = broadcasted(+, a, a)
    @test bc.style === style
    @test similar(bc, elt) isa KroneckerArray{elt, 2, kroneckerfactortypes(a)...}
    @test collect(copy(bc)) ≈ 2 * collect(a)
    bc = broadcasted(*, 2, a)
    @test bc.style === style
    @test collect(copy(bc)) ≈ 2 * collect(a)

    # Mapping
    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    @test_throws "not supported" map(sin, a)
    @test collect(map(Base.Fix1(*, 2), a)) ≈ 2 * collect(a)
    a′ = similar(a)
    @test_throws "not supported" map!(sin, a′, a)
    a′ = similar(a)
    map!(identity, a′, a)
    @test collect(a′) ≈ collect(a)
    a′ = similar(a)
    map!(+, a′, a, a)
    @test collect(a′) ≈ 2 * collect(a)
    a′ = similar(a)
    map!(-, a′, a, a)
    @test norm(collect(a′)) ≈ 0
    a′ = similar(a)
    map!(Base.Fix1(*, 2), a′, a)
    @test collect(a′) ≈ 2 * collect(a)
    a′ = similar(a)
    map!(Base.Fix2(*, 2), a′, a)
    @test collect(a′) ≈ 2 * collect(a)
    a′ = similar(a)
    map!(Base.Fix2(/, 2), a′, a)
    @test collect(a′) ≈ collect(a) / 2
    a′ = similar(a)
    map!(conj, a′, a)
    @test collect(a′) ≈ conj(collect(a))

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    if elt <: Real
        @test real(a) == a
    else
        @test_throws ErrorException real(a)
    end
    if elt <: Real
        @test iszero(imag(a))
    else
        @test_throws ErrorException imag(a)
    end

    # permutedims
    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
    @test permutedims(a, (2, 1, 3)) ==
        permutedims(kroneckerfactors(a, 1), (2, 1, 3)) ⊗ permutedims(kroneckerfactors(a, 2), (2, 1, 3))

    # permutedims!
    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
    b = similar(a)
    permutedims!(b, a, (2, 1, 3))
    @test b == permutedims(kroneckerfactors(a, 1), (2, 1, 3)) ⊗ permutedims(kroneckerfactors(a, 2), (2, 1, 3))

    # Adapt
    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    a′ = adapt(JLArray, a)
    @test a′ isa KroneckerArray{elt, 2, JLArray{elt, 2}, JLArray{elt, 2}}
    @test kroneckerfactors(a′, 1) isa JLArray{elt, 2}
    @test kroneckerfactors(a′, 2) isa JLArray{elt, 2}
    @test Array(kroneckerfactors(a′, 1)) == kroneckerfactors(a, 1)
    @test Array(kroneckerfactors(a′, 2)) == kroneckerfactors(a, 2)

    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
    @test collect(a) ≈ kron_nd(kroneckerfactors(a)...)
    for i in 1:2, j in 1:3, k in 1:2, l in 1:3, m in 1:2, n in 1:3
        @test a[i × j, k × l, m × n] == kroneckerfactors(a, 1)[i, k, m] * kroneckerfactors(a, 2)[j, l, n]
    end
    @test collect(a + a) ≈ 2 * collect(a)
    @test collect(2a) ≈ 2 * collect(a)

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    b = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    c = kroneckerfactors(a, 1) ⊗ kroneckerfactors(b, 2)
    U, S, V = svd(a)
    @test collect(U * diagonal(S) * V') ≈ collect(a)
    @test kroneckerfactors(svdvals(a), 1) ≈ kroneckerfactors(S, 1)
    @test kroneckerfactors(svdvals(a), 2) ≈ kroneckerfactors(S, 2)
    @test sort(collect(S); rev = true) ≈ svdvals(collect(a))
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
        @eval @test_throws ArgumentError $f($a)
    end

    # isapprox

    rng = StableRNG(123)
    a1 = randn(rng, elt, (2, 2))
    a = a1 ⊗ randn(rng, elt, (3, 3))
    b = a1 ⊗ randn(rng, elt, (3, 3))
    @test isapprox(a, b; atol = norm(a - b) * (1 + 2eps(real(elt))))
    @test !isapprox(a, b; atol = norm(a - b) * (1 - 2eps(real(elt))))
    @test isapprox(
        a, b;
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 + 2eps(real(elt)))
    )
    @test !isapprox(
        a, b;
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 - 2eps(real(elt)))
    )
    @test isapprox(
        a, b; atol = norm(a - b) * (1 + 2eps(real(elt))),
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 + 2eps(real(elt)))
    )
    @test isapprox(
        a, b; atol = norm(a - b) * (1 + 2eps(real(elt))),
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 - 2eps(real(elt)))
    )
    @test isapprox(
        a, b; atol = norm(a - b) * (1 - 2eps(real(elt))),
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 + 2eps(real(elt)))
    )
    @test !isapprox(
        a, b; atol = norm(a - b) * (1 - 2eps(real(elt))),
        rtol = norm(a - b) / max(norm(a), norm(b)) * (1 - 2eps(real(elt)))
    )

    a = randn(elt, (2, 2)) ⊗ randn(elt, (3, 3))
    b = randn(elt, (2, 2)) ⊗ randn(elt, (3, 3))
    @test_throws ArgumentError isapprox(a, b)

    # KroneckerArrays.dist_kronecker
    rng = StableRNG(123)
    a = randn(rng, (100, 100))
    b = randn(rng, (100, 100))
    ab = a ⊗ b
    ab′ = (a + randn(rng, size(a)) / 10) ⊗ (b + randn(rng, size(b)) / 10)
    @test KroneckerArrays.dist_kronecker(ab, ab′) ≈ norm(collect(ab) - collect(ab′)) rtol = 1.0e-2
end
