using Adapt: adapt
using DiagonalArrays: δ
using FillArrays: Eye, Zeros
using FunctionImplementations: zero!
using JLArrays: JLArray, jl
using KroneckerArrays: KroneckerArrays, KroneckerArray, ⊗, ×, kroneckerfactors, cartesianrange
using LinearAlgebra: det, norm, pinv
using StableRNGs: StableRNG
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred

@testset "FillArrays.Eye, DiagonalArrays.Delta" begin
    MATRIX_FUNCTIONS = KroneckerArrays.MATRIX_FUNCTIONS
    if VERSION < v"1.11-"
        # `cbrt(::AbstractMatrix{<:Real})` was implemented in Julia 1.11.
        MATRIX_FUNCTIONS = setdiff(MATRIX_FUNCTIONS, [:cbrt])
    end

    a = Eye(2) ⊗ randn(3, 3)
    @test size(a) == (6, 6)
    @test a + a == Eye(2) ⊗ (2 * kroneckerfactors(a, 2))
    @test 2a == Eye(2) ⊗ (2 * kroneckerfactors(a, 2))
    @test a * a == Eye(2) ⊗ (kroneckerfactors(a, 2) * kroneckerfactors(a, 2))
    @test_broken kroneckerfactors(a[(:) × (:), (:) × (:)], 1) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, (:) × (:), (:) × (:)), 1) ≡ Eye(2)
    @test_broken kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)], 1) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:)), 1) ≡ Eye(2)
    @test_broken kroneckerfactors(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)], 1) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:)), 1) ≡ Eye(2)
    @test_broken kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)], 1) ≡
        Eye(2)
    @test_broken kroneckerfactors(
        view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)), 1
    ) ≡ Eye(2)
    @test kroneckerfactors(adapt(JLArray, a), 1) ≡ Eye(2)
    @test kroneckerfactors(adapt(JLArray, a), 2) == jl(kroneckerfactors(a, 2))
    @test kroneckerfactors(adapt(JLArray, a), 2) isa JLArray
    @test_broken kroneckerfactors(similar(a, (cartesianrange(3 × 2), cartesianrange(3 × 2))), 1) ≡ Eye(3)
    @test_broken kroneckerfactors(similar(typeof(a), (cartesianrange(3 × 2), cartesianrange(3 × 2))), 1) ≡
        Eye(3)
    @test_broken kroneckerfactors(similar(a, Float32, (cartesianrange(3 × 2), cartesianrange(3 × 2)))), 1 ≡
        Eye{Float32}(3)
    @test kroneckerfactors(copy(a), 1) ≡ Eye(2)
    @test kroneckerfactors(copy(a), 2) == kroneckerfactors(a, 2)
    b = similar(a)
    @test kroneckerfactors(copyto!(b, a), 1) ≡ Eye(2)
    @test kroneckerfactors(copyto!(b, a), 2) == kroneckerfactors(a, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 1) ≡ Eye(2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 2) == permutedims(kroneckerfactors(a, 2), (2, 1))
    b = similar(a)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 1) ≡ Eye(2)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 2) == permutedims(kroneckerfactors(a, 2), (2, 1))

    a = randn(3, 3) ⊗ Eye(2)
    @test size(a) == (6, 6)
    @test a + a == (2 * kroneckerfactors(a, 1)) ⊗ Eye(2)
    @test 2a == (2 * kroneckerfactors(a, 1)) ⊗ Eye(2)
    @test a * a == (kroneckerfactors(a, 1) * kroneckerfactors(a, 1)) ⊗ Eye(2)
    @test_broken kroneckerfactors(a[(:) × (:), (:) × (:)], 2) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, (:) × (:), (:) × (:)), 2) ≡ Eye(2)
    @test_broken kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)], 2) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:)), 2) ≡ Eye(2)
    @test_broken kroneckerfactors(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)], 2) ≡ Eye(2)
    @test_broken kroneckerfactors(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:)), 2) ≡ Eye(2)
    @test_broken kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)], 2) ≡
        Eye(2)
    @test_broken kroneckerfactors(
        view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)), 2
    ) ≡ Eye(2)
    @test kroneckerfactors(adapt(JLArray, a), 2) ≡ Eye(2)
    @test kroneckerfactors(adapt(JLArray, a), 1) == jl(kroneckerfactors(a, 1))
    @test kroneckerfactors(adapt(JLArray, a), 1) isa JLArray
    @test_broken kroneckerfactors(similar(a, (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡ Eye(3)
    @test_broken kroneckerfactors(similar(typeof(a), (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡
        Eye(3)
    @test_broken kroneckerfactors(similar(a, Float32, (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡
        Eye{Float32}(3)
    @test kroneckerfactors(copy(a), 2) ≡ Eye(2)
    @test kroneckerfactors(copy(a), 2) == kroneckerfactors(a, 2)
    b = similar(a)
    @test kroneckerfactors(copyto!(b, a), 2) ≡ Eye(2)
    @test kroneckerfactors(copyto!(b, a), 2) == kroneckerfactors(a, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 2) ≡ Eye(2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 1) == permutedims(kroneckerfactors(a, 1), (2, 1))
    b = similar(a)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 2) ≡ Eye(2)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 1) == permutedims(kroneckerfactors(a, 1), (2, 1))

    a = δ(2, 2) ⊗ randn(3, 3)
    @test size(a) == (6, 6)
    @test a + a == δ(2, 2) ⊗ (2 * kroneckerfactors(a, 2))
    @test 2a == δ(2, 2) ⊗ (2 * kroneckerfactors(a, 2))
    @test a * a == δ(2, 2) ⊗ (kroneckerfactors(a, 2) * kroneckerfactors(a, 2))
    @test kroneckerfactors(a[(:) × (:), (:) × (:)], 1) ≡ δ(2, 2)
    @test kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)], 1) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:)), 1) ≡ δ(2, 2)
    @test kroneckerfactors(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)], 1) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:)), 1) ≡ δ(2, 2)
    @test kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)], 1) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)), 1) ≡
        δ(2, 2)
    @test kroneckerfactors(adapt(JLArray, a), 1) ≡ δ(2, 2)
    @test kroneckerfactors(adapt(JLArray, a), 2) == jl(kroneckerfactors(a, 2))
    @test kroneckerfactors(adapt(JLArray, a), 2) isa JLArray
    @test_broken kroneckerfactors(similar(a, (cartesianrange(3 × 2), cartesianrange(3 × 2))), 1) ≡ δ(3, 3)
    @test_broken kroneckerfactors(similar(typeof(a), (cartesianrange(3 × 2), cartesianrange(3 × 2))), 1) ≡
        δ(3, 3)
    @test_broken kroneckerfactors(similar(a, Float32, (cartesianrange(3 × 2), cartesianrange(3 × 2))), 1) ≡
        δ(Float32, 3, 3)
    @test kroneckerfactors(copy(a), 1) ≡ δ(2, 2)
    @test kroneckerfactors(copy(a), 2) == kroneckerfactors(a, 2)
    b = similar(a)
    @test kroneckerfactors(copyto!(b, a), 1) ≡ δ(2, 2)
    @test kroneckerfactors(copyto!(b, a), 2) == kroneckerfactors(a, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 1) ≡ δ(2, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 2) == permutedims(kroneckerfactors(a, 2), (2, 1))
    b = similar(a)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 1) ≡ δ(2, 2)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 2) == permutedims(kroneckerfactors(a, 2), (2, 1))

    a = randn(3, 3) ⊗ δ(2, 2)
    @test size(a) == (6, 6)
    @test a + a == (2 * kroneckerfactors(a, 1)) ⊗ δ(2, 2)
    @test 2a == (2 * kroneckerfactors(a, 1)) ⊗ δ(2, 2)
    @test a * a == (kroneckerfactors(a, 1) * kroneckerfactors(a, 1)) ⊗ δ(2, 2)
    @test kroneckerfactors(a[(:) × (:), (:) × (:)], 2) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, (:) × (:), (:) × (:)), 2) ≡ δ(2, 2)
    @test kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), (:) × (:)], 2) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), (:) × (:)), 2) ≡ δ(2, 2)
    @test kroneckerfactors(a[(:) × (:), Base.Slice(Base.OneTo(2)) × (:)], 2) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, (:) × (:), Base.Slice(Base.OneTo(2)) × (:)), 2) ≡ δ(2, 2)
    @test kroneckerfactors(a[Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)], 2) ≡ δ(2, 2)
    @test kroneckerfactors(view(a, Base.Slice(Base.OneTo(2)) × (:), Base.Slice(Base.OneTo(2)) × (:)), 2) ≡
        δ(2, 2)
    @test kroneckerfactors(adapt(JLArray, a), 2) ≡ δ(2, 2)
    @test kroneckerfactors(adapt(JLArray, a), 1) == jl(kroneckerfactors(a, 1))
    @test kroneckerfactors(adapt(JLArray, a), 1) isa JLArray
    @test_broken kroneckerfactors(similar(a, (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡ δ(3, 3)
    @test_broken kroneckerfactors(similar(typeof(a), (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡
        δ(3, 3)
    @test_broken kroneckerfactors(similar(a, Float32, (cartesianrange(2 × 3), cartesianrange(2 × 3))), 2) ≡
        δ(Float32, (3, 3))
    @test kroneckerfactors(copy(a), 2) ≡ δ(2, 2)
    @test kroneckerfactors(copy(a), 2) == kroneckerfactors(a, 2)
    b = similar(a)
    @test kroneckerfactors(copyto!(b, a), 2) ≡ δ(2, 2)
    @test kroneckerfactors(copyto!(b, a), 2) == kroneckerfactors(a, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 2) ≡ δ(2, 2)
    @test kroneckerfactors(permutedims(a, (2, 1)), 1) == permutedims(kroneckerfactors(a, 1), (2, 1))
    b = similar(a)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 2) ≡ δ(2, 2)
    @test kroneckerfactors(permutedims!(b, a, (2, 1)), 1) == permutedims(kroneckerfactors(a, 1), (2, 1))

    # Views
    a = @constinferred(Eye(2) ⊗ randn(3, 3))
    b = @constinferred(view(a, (:) × (2:3), (:) × (2:3)))
    @test_broken kroneckerfactors(b, 1) ≡ Eye(2)
    @test kroneckerfactors(b, 2) ≡ view(kroneckerfactors(a, 2), 2:3, 2:3)
    @test kroneckerfactors(b, 2) == kroneckerfactors(a, 2)[2:3, 2:3]

    a = randn(3, 3) ⊗ Eye(2)
    @test size(a) == (6, 6)
    @test a + a == (2kroneckerfactors(a, 1)) ⊗ Eye(2)
    @test 2a == (2kroneckerfactors(a, 1)) ⊗ Eye(2)
    @test a * a == (kroneckerfactors(a, 1) * kroneckerfactors(a, 1)) ⊗ Eye(2)

    # Views
    a = @constinferred(randn(3, 3) ⊗ Eye(2))
    b = @constinferred(view(a, (2:3) × (:), (2:3) × (:)))
    @test kroneckerfactors(b, 1) ≡ view(kroneckerfactors(a, 1), 2:3, 2:3)
    @test kroneckerfactors(b, 1) == kroneckerfactors(a, 1)[2:3, 2:3]
    @test_broken kroneckerfactors(b, 2) ≡ Eye(2)

    # similar
    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 1) ≡ kroneckerfactors(a, 1)

    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a, eltype(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 1) ≡ kroneckerfactors(a, 1)

    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a, axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 1) ≡ kroneckerfactors(a, 1)

    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a, eltype(a), axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 1) ≡ kroneckerfactors(a, 1)

    @test_broken similar(typeof(a), axes(a))

    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a, Float32)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32, ndims(a)}
    @test_broken kroneckerfactors(a′, 1) ≡ Eye{Float32}(2)

    a = Eye(2) ⊗ randn(3, 3)
    a′ = similar(a, Float32, axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32, ndims(a)}
    @test_broken kroneckerfactors(a′, 1) ≡ Eye{Float32}(2)

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 2) ≡ kroneckerfactors(a, 2)

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a, eltype(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 2) ≡ kroneckerfactors(a, 2)

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a, axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 2) ≡ kroneckerfactors(a, 2)

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a, eltype(a), axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    @test kroneckerfactors(a′, 2) ≡ kroneckerfactors(a, 2)

    @test_broken similar(typeof(a), axes(a))

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a, Float32)
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32, ndims(a)}
    # This is broken because of:
    # https://github.com/JuliaArrays/FillArrays.jl/issues/415
    @test_broken kroneckerfactors(a′, 2) ≡ Eye{Float32}(2)

    a = randn(3, 3) ⊗ Eye(2)
    a′ = similar(a, Float32, axes(a))
    @test size(a′) == (6, 6)
    @test a′ isa KroneckerArray{Float32, ndims(a)}

    a = Eye(3) ⊗ Eye(2)
    for a′ in (
            similar(a), similar(a, eltype(a)), similar(a, axes(a)), similar(a, eltype(a), axes(a)),
        )
        @test size(a′) == (6, 6)
        @test a′ isa KroneckerArray{eltype(a), ndims(a)}
    end
    @test_broken similar(typeof(a), axes(a))

    a = Eye(3) ⊗ Eye(2)
    for args in ((Float32,), (Float32, axes(a)))
        a′ = similar(a, args...)
        @test size(a′) == (6, 6)
        @test a′ isa KroneckerArray{Float32, ndims(a)}
    end

    # FunctionImplementations.zero!
    for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
        zero!(a)
        @test iszero(a)
    end
    a = Eye(3) ⊗ Eye(2)
    @test_throws ErrorException zero!(a)

    # map!(+, ...)
    for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
        a′ = similar(a)
        map!(+, a′, a, a)
        @test collect(a′) ≈ 2 * collect(a)
    end
    a = Eye(3) ⊗ Eye(2)
    a′ = similar(a)
    map!(+, a′, a, a)
    @test a′ ≈ 2a

    # map!(-, ...)
    for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
        a′ = similar(a)
        map!(-, a′, a, a)
        @test norm(collect(a′)) ≈ 0
    end
    a = Eye(3) ⊗ Eye(2)
    a′ = similar(a)
    map!(-, a′, a, a)
    @test iszero(a′)

    # map!(-, b, a)
    for a in (Eye(2) ⊗ randn(3, 3), randn(3, 3) ⊗ Eye(2))
        a′ = similar(a)
        map!(-, a′, a)
        @test collect(a′) ≈ -collect(a)
    end
    a = Eye(3) ⊗ Eye(2)
    a′ = similar(a)
    map!(-, a′, a)
    @test a′ ≈ -a

    ## # Eye ⊗ A
    ## rng = StableRNG(123)
    ## a = Eye(2) ⊗ randn(rng, 3, 3)
    ## for f in MATRIX_FUNCTIONS
    ##   @eval begin
    ##     fa = $f($a)
    ##     @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
    ##     @test kroneckerfactors(fa) isa Eye
    ##   end
    ## end

    fa = inv(a)
    @test collect(fa) ≈ inv(collect(a))
    @test kroneckerfactors(fa, 1) isa Eye

    fa = pinv(a)
    @test collect(fa) ≈ pinv(collect(a))
    @test_broken kroneckerfactors(fa, 1) isa Eye

    @test det(a) ≈ det(collect(a))

    ## # A ⊗ Eye
    ## rng = StableRNG(123)
    ## a = randn(rng, 3, 3) ⊗ Eye(2)
    ## for f in setdiff(MATRIX_FUNCTIONS, [:atanh])
    ##   @eval begin
    ##     fa = $f($a)
    ##     @test collect(fa) ≈ $f(collect($a)) rtol = ∜(eps(real(eltype($a))))
    ##     @test kroneckerfactors(fa) isa Eye
    ##   end
    ## end

    fa = inv(a)
    @test collect(fa) ≈ inv(collect(a))
    @test kroneckerfactors(fa, 2) isa Eye

    fa = pinv(a)
    @test collect(fa) ≈ pinv(collect(a))
    @test_broken kroneckerfactors(fa, 2) isa Eye

    @test det(a) ≈ det(collect(a))

    # Eye ⊗ Eye
    a = Eye(2) ⊗ Eye(2)
    for f in MATRIX_FUNCTIONS
        @eval begin
            @test $f($a) == kroneckerfactors($a, 1) ⊗ $f(kroneckerfactors($a, 2))
        end
    end

    fa = inv(a)
    @test fa == a
    @test kroneckerfactors(fa, 1) isa Eye
    @test kroneckerfactors(fa, 2) isa Eye

    fa = pinv(a)
    @test fa == a
    @test_broken kroneckerfactors(fa, 1) isa Eye
    @test_broken kroneckerfactors(fa, 2) isa Eye

    @test det(a) ≈ det(collect(a)) ≈ 1

    # permutedims
    a = Eye(2, 2) ⊗ randn(3, 3)
    @test permutedims(a, (2, 1)) == Eye(2, 2) ⊗ permutedims(kroneckerfactors(a, 2), (2, 1))

    a = randn(2, 2) ⊗ Eye(3, 3)
    @test permutedims(a, (2, 1)) == permutedims(kroneckerfactors(a, 1), (2, 1)) ⊗ Eye(3, 3)

    # permutedims!
    a = Eye(2, 2) ⊗ randn(3, 3)
    b = similar(a)
    permutedims!(b, a, (2, 1))
    @test b == Eye(2, 2) ⊗ permutedims(kroneckerfactors(a, 2), (2, 1))

    a = randn(3, 3) ⊗ Eye(2, 2)
    b = similar(a)
    permutedims!(b, a, (2, 1))
    @test b == permutedims(kroneckerfactors(a, 1), (2, 1)) ⊗ Eye(2, 2)
end

@testset "FillArrays.Zeros" begin
    a = randn(2, 2) ⊗ randn(2, 2)
    b = Zeros(2, 2) ⊗ Zeros(2, 2)
    for (x, y) in ((a, b), (b, a))
        @test x + y == a
        @test x .+ y == a
        @test map!(+, similar(a), x, y) == a
        @test (similar(a) .= x .+ y) == a
    end

    @test a - b == a
    @test a .- b == a
    @test map!(-, similar(a), a, b) == a
    @test (similar(a) .= a .- b) == a

    @test b - a == -a
    @test b .- a == -a
    @test map!(-, similar(a), b, a) == -a
    @test (similar(a) .= b .- a) == -a

    @test b + b == b
    @test b .+ b == b
    @test b - b == b
    @test b .- b == b
end
