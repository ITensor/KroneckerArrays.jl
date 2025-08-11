module KroneckerArraysTensorProductsExt

using KroneckerArrays: CartesianProductOneTo, ×, arg1, arg2, cartesianrange, unproduct
using TensorProducts: TensorProducts, tensor_product
function TensorProducts.tensor_product(a1::CartesianProductOneTo, a2::CartesianProductOneTo)
  prod = tensor_product(arg1(a1), arg1(a2)) × tensor_product(arg2(a1), arg2(a2))
  range = tensor_product(unproduct(a1), unproduct(a2))
  return cartesianrange(prod, range)
end

end
