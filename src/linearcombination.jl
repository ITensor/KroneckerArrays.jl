using Base.Broadcast: Broadcasted
struct LinearCombination{C} <: Function
  coefficients::C
end
coefficients(a::LinearCombination) = a.coefficients
function (f::LinearCombination)(args...)
  return mapreduce(*,+,coefficients(f),args)
end

struct Sum{Style,C<:Tuple,A<:Tuple}
  style::Style
  coefficients::C
  arguments::A
end
coefficients(a::Sum) = a.coefficients
arguments(a::Sum) = a.arguments
style(a::Sum) = a.style
LinearCombination(a::Sum) = LinearCombination(coefficients(a))
using Base.Broadcast: combine_axes
Base.axes(a::Sum) = combine_axes(a.arguments...)
function Base.eltype(a::Sum)
  cts = typeof.(coefficients(a))
  elts = eltype.(arguments(a))
  ts = map((ct, elt) -> Base.promote_op(*, ct, elt), cts, elts)
  return Base.promote_op(+, ts...)
end
using Base.Broadcast: combine_styles
function Sum(coefficients::Tuple, arguments::Tuple)
  return Sum(combine_styles(arguments...), coefficients, arguments)
end
Sum(a) = Sum((one(eltype(a)),), (a,))
function Base.:+(a::Sum, b::Sum)
  return Sum((coefficients(a)..., coefficients(b)...), (arguments(a)..., arguments(b)...))
end
Base.:-(a::Sum, b::Sum) = a + (-b)
Base.:+(a::Sum, b::AbstractArray) = a + Sum(b)
Base.:-(a::Sum, b::AbstractArray) = a - Sum(b)
Base.:+(a::AbstractArray, b::Sum) = Sum(a) + b
Base.:-(a::AbstractArray, b::Sum) = Sum(a) - b
Base.:*(c::Number, a::Sum) = Sum(c .* coefficients(a), arguments(a))
Base.:*(a::Sum, c::Number) = c * a
Base.:/(a::Sum, c::Number) = Sum(coefficients(a) ./ c, arguments(a))
Base.:-(a::Sum) = -1 * a

function Base.copy(a::Sum)
  return copyto!(similar(a), a)
end
Base.similar(a::Sum) = similar(a, eltype(a))
Base.similar(a::Sum, elt::Type) = similar(a, elt, axes(a))
function Base.copyto!(dest::AbstractArray, a::Sum)
  f = LinearCombination(a)
  dest .= f.(arguments(a)...)
  return dest
end
function Broadcast.Broadcasted(a::Sum)
  f = LinearCombination(a)
  return Broadcasted(style(a), f, arguments(a), axes(a))
end
function Base.similar(a::Sum, elt::Type, ax::Tuple)
  return similar(Broadcasted(a), elt, ax)
end

using Base.Broadcast: Broadcast, AbstractArrayStyle, DefaultArrayStyle
Broadcast.materialize(a::Sum) = copy(a)
Broadcast.materialize!(dest, a::Sum) = copyto!(dest, a)
struct SumStyle <: AbstractArrayStyle{Any} end
Broadcast.broadcastable(a::Sum) = a
Broadcast.BroadcastStyle(::Type{<:Sum}) = SumStyle()
Broadcast.BroadcastStyle(style::SumStyle, ::AbstractArrayStyle) = style
# Fix ambiguity error with Base.
Broadcast.BroadcastStyle(style::SumStyle, ::DefaultArrayStyle) = style
function Broadcast.broadcasted(::SumStyle, f, as...)
  return error("Arbitrary broadcasting not supported for SumStyle.")
end
function Broadcast.broadcasted(::SumStyle, ::typeof(+), a, b::Sum)
  return Sum(a) + b
end
function Broadcast.broadcasted(::SumStyle, ::typeof(+), a::Sum, b)
  return a + Sum(b)
end
function Broadcast.broadcasted(::SumStyle, ::typeof(+), a::Sum, b::Sum)
  return a + b
end
function Broadcast.broadcasted(::SumStyle, ::typeof(*), c::Number, a)
  return c * Sum(a)
end
function Broadcast.broadcasted(::SumStyle, ::typeof(*), c::Number, a::Sum)
  return c * a
end
function Broadcast.broadcasted(::SumStyle, ::typeof(/), a::Sum, c::Number)
  return Sum(a) / c
end
