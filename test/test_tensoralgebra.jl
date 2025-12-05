using TensorAlgebra: matricize, tensor_product_axis, unmatricize
using KroneckerArrays: ⊗, cartesianrange, kroneckerfactors, unproduct
using Test: @test, @testset

@testset "TensorAlgebraExt" begin
    @testset "tensor_product_axis" begin
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
        @test m == matricize(kroneckerfactors(a, 1), (1, 2), (3,)) ⊗ matricize(kroneckerfactors(a, 2), (1, 2), (3,))
        @test unmatricize(m, (axes(a, 1), axes(a, 2)), (axes(a, 3),)) == a
    end
end
