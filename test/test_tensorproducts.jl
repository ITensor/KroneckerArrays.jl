using KroneckerArrays: ×, arg1, arg2, cartesianrange, unproduct
using TensorProducts: tensor_product
using Test: @test, @testset

@testset "KroneckerArraysTensorProductsExt" begin
  r1 = cartesianrange(2, 3)
  r2 = cartesianrange(4, 5)
  r = tensor_product(r1, r2)
  @test r ≡ cartesianrange(8, 15)
  @test arg1(r) ≡ Base.OneTo(8)
  @test arg2(r) ≡ Base.OneTo(15)
  @test unproduct(r) ≡ Base.OneTo(120)
end
