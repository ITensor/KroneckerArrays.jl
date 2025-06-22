struct CartesianPair{A,B}
  a::A
  b::B
end
arguments(a::CartesianPair) = (a.a, a.b)
arguments(a::CartesianPair, n::Int) = arguments(a)[n]

arg1(a::CartesianPair) = a.a
arg2(a::CartesianPair) = a.b

×(a, b) = CartesianPair(a, b)

function Base.show(io::IO, a::CartesianPair)
  print(io, a.a, " × ", a.b)
  return nothing
end

struct CartesianProduct{TA,TB,A<:AbstractVector{TA},B<:AbstractVector{TB}} <:
       AbstractVector{CartesianPair{TA,TB}}
  a::A
  b::B
end
arguments(a::CartesianProduct) = (a.a, a.b)
arguments(a::CartesianProduct, n::Int) = arguments(a)[n]

arg1(a::CartesianProduct) = a.a
arg2(a::CartesianProduct) = a.b

function Base.show(io::IO, a::CartesianProduct)
  print(io, a.a, " × ", a.b)
  return nothing
end

# This is used when printing block sparse arrays with KroneckerArray
# blocks.
# TODO: Investigate if this is needed or if it can be avoided
# by iterating over CartesianProduct axes.
function Base.checkindex(::Type{Bool}, inds::CartesianProduct, i::Int)
  return checkindex(Bool, Base.OneTo(length(inds)), i)
end

×(a::AbstractVector, b::AbstractVector) = CartesianProduct(a, b)
Base.length(a::CartesianProduct) = length(arg1(a)) * length(arg2(a))
Base.size(a::CartesianProduct) = (length(a),)
function Base.getindex(a::CartesianProduct, i::CartesianProduct)
  return arg1(a)[arg1(i)] × arg2(a)[arg2(i)]
end
function Base.getindex(a::CartesianProduct, i::CartesianPair)
  return arg1(a)[arg1(i)] × arg2(a)[arg2(i)]
end
function Base.getindex(a::CartesianProduct, i::Int)
  I = Tuple(CartesianIndices((length(arg1(a)), length(arg2(a))))[i])
  return a[I[1] × I[2]]
end

using Base: promote_shape
function Base.promote_shape(
  a::Tuple{Vararg{CartesianProduct}}, b::Tuple{Vararg{CartesianProduct}}
)
  return promote_shape(arg1.(a), arg1.(b)) × promote_shape(arg2.(a), arg2.(b))
end

using Base.Broadcast: axistype
function Base.Broadcast.axistype(r1::CartesianProduct, r2::CartesianProduct)
  return axistype(arg1(r1), arg1(r2)) × axistype(arg2(r1), arg2(r2))
end

## function Base.to_index(A::KroneckerArray, I::CartesianProduct)
##   return I
## end

struct CartesianProductUnitRange{T,P<:CartesianProduct,R<:AbstractUnitRange{T}} <:
       AbstractUnitRange{T}
  product::P
  range::R
end
Base.first(r::CartesianProductUnitRange) = first(r.range)
Base.last(r::CartesianProductUnitRange) = last(r.range)

cartesianproduct(r::CartesianProductUnitRange) = getfield(r, :product)
unproduct(r::CartesianProductUnitRange) = getfield(r, :range)

arg1(a::CartesianProductUnitRange) = arg1(cartesianproduct(a))
arg2(a::CartesianProductUnitRange) = arg2(cartesianproduct(a))

function Base.show(io::IO, r::CartesianProductUnitRange)
  print(io, cartesianproduct(r), ": ", unproduct(r))
  return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", r::CartesianProductUnitRange)
  show(io, mime, cartesianproduct(r))
  println(io)
  show(io, mime, unproduct(r))
  return nothing
end

function CartesianProductUnitRange(p::CartesianProduct)
  return CartesianProductUnitRange(p, Base.OneTo(length(p)))
end
function CartesianProductUnitRange(a, b)
  return CartesianProductUnitRange(a × b)
end
to_product_indices(a::AbstractVector) = a
to_product_indices(i::Integer) = Base.OneTo(i)
cartesianrange(a, b) = cartesianrange(to_product_indices(a) × to_product_indices(b))
function cartesianrange(p::CartesianProduct)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return cartesianrange(p′, Base.OneTo(length(p′)))
end
function cartesianrange(p::CartesianProduct, range::AbstractUnitRange)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return CartesianProductUnitRange(p′, range)
end

Base.axes(r::CartesianProductUnitRange) = (cartesianrange(cartesianproduct(r)),)

using Base.Broadcast: DefaultArrayStyle
for f in (:+, :-)
  @eval begin
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), r::CartesianProductUnitRange, x::Integer
    )
      return CartesianProductUnitRange(r.product, $f.(r.range, x))
    end
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), x::Integer, r::CartesianProductUnitRange
    )
      return CartesianProductUnitRange(r.product, $f.(x, r.range))
    end
  end
end

using Base.Broadcast: axistype
function Base.Broadcast.axistype(
  r1::CartesianProductUnitRange, r2::CartesianProductUnitRange
)
  prod = axistype(arg1(r1), arg1(r2)) × axistype(arg2(r1), arg2(r2))
  range = axistype(unproduct(r1), unproduct(r2))
  return cartesianrange(prod, range)
end

function Base.checkindex(::Type{Bool}, inds::CartesianProductUnitRange, i::CartesianPair)
  return checkindex(Bool, arg1(inds), arg1(i)) && checkindex(Bool, arg2(inds), arg2(i))
end
