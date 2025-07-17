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

Base.copy(a::CartesianProduct) = copy(arg1(a)) × copy(arg2(a))

function Base.show(io::IO, a::CartesianProduct)
  print(io, a.a, " × ", a.b)
  return nothing
end
function Base.show(io::IO, ::MIME"text/plain", a::CartesianProduct)
  show(io, a)
  return nothing
end

×(a::AbstractVector, b::AbstractVector) = CartesianProduct(a, b)
Base.length(a::CartesianProduct) = length(a.a) * length(a.b)
Base.size(a::CartesianProduct) = (length(a),)

function Base.getindex(a::CartesianProduct, i::CartesianProduct)
  return arg1(a)[arg1(i)] × arg2(a)[arg2(i)]
end
function Base.getindex(a::CartesianProduct, i::CartesianPair)
  return arg1(a)[arg1(i)] × arg2(a)[arg2(i)]
end
function Base.getindex(a::CartesianProduct, i::Int)
  I = Tuple(CartesianIndices((length(arg2(a)), length(arg1(a))))[i])
  return a[I[2] × I[1]]
end

struct CartesianProductVector{T,P<:CartesianProduct,V<:AbstractVector{T}} <:
       AbstractVector{T}
  product::P
  values::V
end
cartesianproduct(r::CartesianProductVector) = getfield(r, :product)
unproduct(r::CartesianProductVector) = getfield(r, :values)
Base.length(a::CartesianProductVector) = length(unproduct(a))
Base.size(a::CartesianProductVector) = (length(a),)
function Base.axes(r::CartesianProductVector)
  return (CartesianProductUnitRange(cartesianproduct(r), only(axes(unproduct(r)))),)
end
function Base.copy(a::CartesianProductVector)
  return CartesianProductVector(copy(cartesianproduct(a)), copy(unproduct(a)))
end
function Base.getindex(r::CartesianProductVector, i::Integer)
  return unproduct(r)[i]
end

function Base.show(io::IO, a::CartesianProductVector)
  show(io, unproduct(a))
  return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::CartesianProductVector)
  show(io, mime, cartesianproduct(a))
  println(io)
  show(io, mime, unproduct(a))
  return nothing
end

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

function Base.show(io::IO, a::CartesianProductUnitRange)
  show(io, unproduct(a))
  return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::CartesianProductUnitRange)
  show(io, mime, cartesianproduct(a))
  println(io)
  show(io, mime, unproduct(a))
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
function cartesianrange(p::CartesianPair)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return cartesianrange(p′)
end
function cartesianrange(p::CartesianProduct)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return cartesianrange(p′, Base.OneTo(length(p′)))
end
function cartesianrange(p::CartesianPair, range::AbstractUnitRange)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return cartesianrange(p′, range)
end
function cartesianrange(p::CartesianProduct, range::AbstractUnitRange)
  p′ = to_product_indices(arg1(p)) × to_product_indices(arg2(p))
  return CartesianProductUnitRange(p′, range)
end

function Base.axes(r::CartesianProductUnitRange)
  prod = cartesianproduct(r)
  prod_ax = only(axes(arg1(prod))) × only(axes(arg2(prod)))
  return (CartesianProductUnitRange(prod_ax, only(axes(unproduct(r)))),)
end

function Base.checkindex(::Type{Bool}, inds::CartesianProductUnitRange, i::CartesianPair)
  return checkindex(Bool, arg1(inds), arg1(i)) && checkindex(Bool, arg2(inds), arg2(i))
end

const CartesianProductOneTo{T,P<:CartesianProduct,R<:Base.OneTo{T}} = CartesianProductUnitRange{
  T,P,R
}
Base.axes(S::Base.Slice{<:CartesianProductOneTo}) = (S.indices,)
Base.axes1(S::Base.Slice{<:CartesianProductOneTo}) = S.indices
Base.unsafe_indices(S::Base.Slice{<:CartesianProductOneTo}) = (S.indices,)

function Base.getindex(a::CartesianProductUnitRange, I::CartesianProduct)
  prod = cartesianproduct(a)
  prod_I = arg1(prod)[arg1(I)] × arg2(prod)[arg2(I)]
  return CartesianProductVector(prod_I, map(Base.Fix1(getindex, a), I))
end

# Reverse map from CartesianPair to linear index in the range.
function Base.getindex(inds::CartesianProductUnitRange, i::CartesianPair)
  i′ = (findfirst(==(arg2(i)), arg2(inds)), findfirst(==(arg1(i)), arg1(inds)))
  return inds[LinearIndices((length(arg2(inds)), length(arg1(inds))))[i′...]]
end

using Base.Broadcast: DefaultArrayStyle
for f in (:+, :-)
  @eval begin
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), r::CartesianProductUnitRange, x::Integer
    )
      return CartesianProductUnitRange(cartesianproduct(r), $f.(unproduct(r), x))
    end
    function Broadcast.broadcasted(
      ::DefaultArrayStyle{1}, ::typeof($f), x::Integer, r::CartesianProductUnitRange
    )
      return CartesianProductUnitRange(cartesianproduct(r), $f.(x, unproduct(r)))
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
