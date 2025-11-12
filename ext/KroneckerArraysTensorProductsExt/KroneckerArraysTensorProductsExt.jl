module KroneckerArraysTensorProductsExt

using TensorProducts: TensorProducts, tensor_product
using KroneckerArrays: CartesianProductOneTo, kroneckerfactors, cartesianrange, unproduct

function TensorProducts.tensor_product(a1::CartesianProductOneTo, a2::CartesianProductOneTo)
    return cartesianrange(
        tensor_product(kroneckerfactors(a1, 1), kroneckerfactors(a2, 1)),
        tensor_product(kroneckerfactors(a1, 2), kroneckerfactors(a2, 2)),
        tensor_product(unproduct(a1), unproduct(a2))
    )
end

end
