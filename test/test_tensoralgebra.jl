using TensorAlgebra: FusionStyle, matricize, tensor_product_axis, trivial_axis, unmatricize
using KroneckerArrays: ⊗, cartesianrange, kroneckerfactors, unproduct
using Test: @test, @testset

@testset "TensorAlgebraExt" begin
    @testset "FusionStyle" begin
        a1 = randn(2, 2, 2)
        a2 = randn(2, 2, 2)
        a = a1 ⊗ a2
        @test FusionStyle(a) ≡ ReshapeFusion() ⊗ ReshapeFusion()
        @test kroneckerfactors(FusionStyle(a)) ≡ (ReshapeFusion(), ReshapeFusion())
        @test typeof(FusionStyle(a))() ≡ FusionStyle(a)
    end
    @testset "tensor_product_axis" begin
        r = cartesianrange(2, 3)
        @test trivial_axis(r) ≡ cartesianrange(1, 1)

        r1 = cartesianrange(2, 3)
        r2 = cartesianrange(4, 5)
        r = tensor_product_axis(r1, r2)
        @test r ≡ cartesianrange(8, 15)
        @test kroneckerfactors(r, 1) ≡ Base.OneTo(8)
        @test kroneckerfactors(r, 2) ≡ Base.OneTo(15)
        @test unproduct(r) ≡ Base.OneTo(120)
    end
    @testset "matricize/unmatricize" begin
        a = randn(2, 2, 2) ⊗ randn(3, 3, 3)
        m = matricize(a, (1, 2), (3,))
        @test m == matricize(kroneckerfactors(a, 1), (1, 2), (3,)) ⊗
            matricize(kroneckerfactors(a, 2), (1, 2), (3,))
        @test unmatricize(m, (axes(a, 1), axes(a, 2)), (axes(a, 3),)) == a
    end
end
