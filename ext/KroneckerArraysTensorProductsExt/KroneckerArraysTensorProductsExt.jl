module KroneckerArraysTensorProductsExt

using KroneckerArrays: CartesianProductOneTo, ×, arg1, arg2, cartesianrange
using TensorProducts: TensorProducts, tensor_product
function TensorProducts.tensor_product(a1::CartesianProductOneTo, a2::CartesianProductOneTo)
  return cartesianrange(
    tensor_product(arg1(a1), arg1(a2)) × tensor_product(arg2(a1), arg2(a2))
  )
end

end
