using TensorAlgebra: matricize, unmatricize
using KroneckerArrays: ⊗, kroneckerfactors
using Test: @test, @testset

@testset "TensorAlgebraExt" begin
    a = randn(2, 2, 2) ⊗ randn(3, 3, 3)
    m = matricize(a, (1, 2), (3,))
    @test m == matricize(kroneckerfactors(a, 1), (1, 2), (3,)) ⊗ matricize(kroneckerfactors(a, 2), (1, 2), (3,))
    @test unmatricize(m, (axes(a, 1), axes(a, 2)), (axes(a, 3),)) == a
end
