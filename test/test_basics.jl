using Adapt: adapt
using Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted
using DerivableInterfaces: zero!
using DiagonalArrays: diagonal
using GPUArraysCore: @allowscalar
using JLArrays: JLArray
using KroneckerArrays: KroneckerArrays, KroneckerArray, KroneckerStyle,
    CartesianProductUnitRange, CartesianProductVector, ⊗, ×, arg1, arg2, cartesianproduct,
    cartesianrange, kron_nd, unproduct
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
        @constinferred(cartesianrange(Base.OneTo(2) × Base.OneTo(3)))
    @test @constinferred(cartesianproduct(r)) === Base.OneTo(2) × Base.OneTo(3)
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

    @test sprint(show, "text/plain", cartesianrange(2 × 3)) ==
        "Base.OneTo(2) × Base.OneTo(3)\nBase.OneTo(6)"
    @test sprint(show, cartesianrange(2 × 3)) == "Base.OneTo(6)"

    # CartesianProductUnitRange axes
    r = cartesianrange((2:3) × (3:4), 2:5)
    @test axes(r) ≡ (CartesianProductUnitRange(Base.OneTo(2) × Base.OneTo(2), Base.OneTo(4)),)

    # CartesianProductUnitRange getindex
    r1 = cartesianrange((2:4) × (3:5), 2:10)
    r2 = cartesianrange((2:3) × (2:3), 2:5)
    @test r1[r2] ≡ cartesianrange((3:4) × (4:5), 3:6)

    @test axes(r) ≡ (CartesianProductUnitRange(Base.OneTo(2) × Base.OneTo(2), Base.OneTo(4)),)

    # CartesianProductVector axes
    r = CartesianProductVector(([2, 4]) × ([3, 5]), [3, 5, 7, 9])
    @test axes(r) ≡ (CartesianProductUnitRange(Base.OneTo(2) × Base.OneTo(2), Base.OneTo(4)),)

    r = @constinferred(cartesianrange(2 × 3, 2:7))
    @test r === cartesianrange(Base.OneTo(2) × Base.OneTo(3), 2:7)
    @test cartesianproduct(r) === Base.OneTo(2) × Base.OneTo(3)
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
    c = @constinferred(a.arg1 ⊗ b.arg2)
    @test a isa KroneckerArray{elt, 2, typeof(a.arg1), typeof(a.arg2)}
    @test similar(typeof(a), (2, 3)) isa Matrix{elt}
    @test size(similar(typeof(a), (2, 3))) == (2, 3)
    @test isreal(a) == (elt <: Real)
    @test a[1 × 1, 1 × 1] == a.arg1[1, 1] * a.arg2[1, 1]
    @test a[1 × 3, 2 × 1] == a.arg1[1, 2] * a.arg2[3, 1]
    @test a[1 × (2:3), 2 × 1] == a.arg1[1, 2] * a.arg2[2:3, 1]
    @test a[1 × :, (:) × 1] == a.arg1[1, :] ⊗ a.arg2[:, 1]
    @test a[(1:2) × (2:3), (1:2) × (2:3)] == a.arg1[1:2, 1:2] ⊗ a.arg2[2:3, 2:3]
    v = randn(elt, 2) ⊗ randn(elt, 3)
    @test v[1 × 1] == v.arg1[1] * v.arg2[1]
    @test v[1 × 3] == v.arg1[1] * v.arg2[3]
    @test v[(1:2) × 3] == v.arg1[1:2] * v.arg2[3]
    @test v[(1:2) × (2:3)] == v.arg1[1:2] ⊗ v.arg2[2:3]
    @test eltype(a) === elt
    @test collect(a) == kron(collect(a.arg1), collect(a.arg2))
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
    @test arg1(b) === view(arg1(a), 1:2, 1:2)
    @test arg1(b) == arg1(a)[1:2, 1:2]
    @test arg2(b) === view(arg2(a), 2:3, 2:3)
    @test arg2(b) == arg2(a)[2:3, 2:3]

    # Broadcasting
    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    style = KroneckerStyle(BroadcastStyle(typeof(a.arg1)), BroadcastStyle(typeof(a.arg2)))
    @test BroadcastStyle(typeof(a)) === style
    @test_throws "not supported" sin.(a)
    a′ = similar(a)
    @test_throws "not supported" a′ .= sin.(a)
    a′ = similar(a)
    a′ .= 2 .* a
    @test collect(a′) ≈ 2 * collect(a)
    bc = broadcasted(+, a, a)
    @test bc.style === style
    @test similar(bc, elt) isa KroneckerArray{elt, 2, typeof(a.arg1), typeof(a.arg2)}
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
        permutedims(arg1(a), (2, 1, 3)) ⊗ permutedims(arg2(a), (2, 1, 3))

    # permutedims!
    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
    b = similar(a)
    permutedims!(b, a, (2, 1, 3))
    @test b == permutedims(arg1(a), (2, 1, 3)) ⊗ permutedims(arg2(a), (2, 1, 3))

    # Adapt
    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    a′ = adapt(JLArray, a)
    @test a′ isa KroneckerArray{elt, 2, JLArray{elt, 2}, JLArray{elt, 2}}
    @test a′.arg1 isa JLArray{elt, 2}
    @test a′.arg2 isa JLArray{elt, 2}
    @test Array(a′.arg1) == a.arg1
    @test Array(a′.arg2) == a.arg2

    a = randn(elt, 2, 2, 2) ⊗ randn(elt, 3, 3, 3)
    @test collect(a) ≈ kron_nd(a.arg1, a.arg2)
    @test a[1 × 1, 1 × 1, 1 × 1] == a.arg1[1, 1, 1] * a.arg2[1, 1, 1]
    @test a[1 × 3, 2 × 1, 2 × 2] == a.arg1[1, 2, 2] * a.arg2[3, 1, 2]
    @test collect(a + a) ≈ 2 * collect(a)
    @test collect(2a) ≈ 2 * collect(a)

    a = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    b = randn(elt, 2, 2) ⊗ randn(elt, 3, 3)
    c = arg1(a) ⊗ arg2(b)
    U, S, V = svd(a)
    @test collect(U * diagonal(S) * V') ≈ collect(a)
    @test arg1(svdvals(a)) ≈ arg1(S)
    @test arg2(svdvals(a)) ≈ arg2(S)
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
        @eval begin
            @test_throws ArgumentError $f($a)
        end
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
    a = randn(rng, (100, 100)) ⊗ randn(rng, (100, 100))
    b = (arg1(a) + randn(rng, size(arg1(a))) / 10) ⊗
        (arg2(a) + randn(rng, size(arg2(a))) / 10)
    @test KroneckerArrays.dist_kronecker(a, b) ≈ norm(collect(a) - collect(b)) rtol = 1.0e-2
end
