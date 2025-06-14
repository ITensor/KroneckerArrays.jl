struct CartesianProduct{A,B}
  a::A
  b::B
end
arguments(a::CartesianProduct) = (a.a, a.b)
arguments(a::CartesianProduct, n::Int) = arguments(a)[n]

function Base.show(io::IO, a::CartesianProduct)
  print(io, a.a, " × ", a.b)
  return nothing
end

×(a, b) = CartesianProduct(a, b)
Base.length(a::CartesianProduct) = length(a.a) * length(a.b)
Base.getindex(a::CartesianProduct, i::CartesianProduct) = a.a[i.a] × a.b[i.b]

function Base.iterate(a::CartesianProduct, state...)
  x = iterate(Iterators.product(a.a, a.b), state...)
  isnothing(x) && return x
  next, new_state = x
  return ×(next...), new_state
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

function CartesianProductUnitRange(p::CartesianProduct)
  return CartesianProductUnitRange(p, Base.OneTo(length(p)))
end
function CartesianProductUnitRange(a, b)
  return CartesianProductUnitRange(a × b)
end
to_range(a::AbstractUnitRange) = a
to_range(i::Integer) = Base.OneTo(i)
cartesianrange(a, b) = cartesianrange(to_range(a) × to_range(b))
function cartesianrange(p::CartesianProduct)
  p′ = to_range(p.a) × to_range(p.b)
  return cartesianrange(p′, Base.OneTo(length(p′)))
end
function cartesianrange(p::CartesianProduct, range::AbstractUnitRange)
  p′ = to_range(p.a) × to_range(p.b)
  return CartesianProductUnitRange(p′, range)
end

function Base.axes(r::CartesianProductUnitRange)
  return (CartesianProductUnitRange(r.product, only(axes(r.range))),)
end

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
