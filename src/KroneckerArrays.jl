module KroneckerArrays

export ⊗, ×

struct CartesianProduct{A,B}
  a::A
  b::B
end
arguments(a::CartesianProduct) = (a.a, a.b)
arguments(a::CartesianProduct, n::Int) = arguments(a)[n]

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

struct KroneckerArray{T,N,A<:AbstractArray{T,N},B<:AbstractArray{T,N}} <: AbstractArray{T,N}
  a::A
  b::B
end
function KroneckerArray(a::AbstractArray, b::AbstractArray)
  if ndims(a) != ndims(b)
    throw(
      ArgumentError("Kronecker product requires arrays of the same number of dimensions.")
    )
  end
  return KroneckerArray(Base.promote_eltype(a, b)...)
end
const KroneckerMatrix{T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}} = KroneckerArray{T,2,A,B}
const KroneckerVector{T,A<:AbstractVector{T},B<:AbstractVector{T}} = KroneckerArray{T,1,A,B}

function Base.similar(
  a::AbstractArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(a, elt, map(ax -> ax.product.a, axs)) ⊗
         similar(a, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  a::KroneckerArray,
  elt::Type,
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(a.a, elt, map(ax -> ax.product.a, axs)) ⊗
         similar(a.b, elt, map(ax -> ax.product.b, axs))
end
function Base.similar(
  arrayt::Type{<:AbstractArray},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
)
  return similar(arrayt, map(ax -> ax.product.a, axs)) ⊗
         similar(arrayt, map(ax -> ax.product.b, axs))
end

Base.collect(a::KroneckerArray) = kron(a.a, a.b)

Base.size(a::KroneckerArray) = ntuple(dim -> size(a.a, dim) * size(a.b, dim), ndims(a))

function Base.axes(a::KroneckerArray)
  return ntuple(ndims(a)) do dim
    return CartesianProductUnitRange(
      axes(a.a, dim) × axes(a.b, dim), Base.OneTo(size(a, dim))
    )
  end
end

arguments(a::KroneckerArray) = (a.a, a.b)
arguments(a::KroneckerArray, n::Int) = arguments(a)[n]

function Base.print_array(io::IO, a::KroneckerArray)
  Base.print_array(io, a.a)
  println(io, "\n ⊗")
  Base.print_array(io, a.b)
  return nothing
end
function Base.show(io::IO, a::KroneckerArray)
  show(io, a.a)
  print(io, " ⊗ ")
  show(io, a.b)
  return nothing
end

⊗(a::AbstractVecOrMat, b::AbstractVecOrMat) = KroneckerArray(a, b)
⊗(a::Number, b::Number) = a * b
⊗(a::Number, b::AbstractVecOrMat) = a * b
⊗(a::AbstractVecOrMat, b::Number) = a * b

function Base.getindex(::KroneckerArray, ::Int)
  return throw(ArgumentError("Scalar indexing of KroneckerArray is not supported."))
end
function Base.getindex(::KroneckerArray{<:Any,N}, ::Vararg{Int,N}) where {N}
  return throw(ArgumentError("Scalar indexing of KroneckerArray is not supported."))
end
function Base.getindex(a::KroneckerVector, i::CartesianProduct)
  return a.a[i.a] ⊗ a.b[i.b]
end
function Base.getindex(a::KroneckerMatrix, i::CartesianProduct, j::CartesianProduct)
  return a.a[i.a, j.a] ⊗ a.b[i.b, j.b]
end

function Base.:(==)(a::KroneckerArray, b::KroneckerArray)
  return a.a == b.a && a.b == b.b
end
function Base.iszero(a::KroneckerArray)
  return iszero(a.a) || iszero(a.b)
end
function Base.inv(a::KroneckerArray)
  return inv(a.a) ⊗ inv(a.b)
end
function Base.transpose(a::KroneckerArray)
  return transpose(a.a) ⊗ transpose(a.b)
end
function Base.adjoint(a::KroneckerArray)
  return a.a' ⊗ a.b'
end

function Base.:*(a::Number, b::KroneckerArray)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::KroneckerArray, b::Number)
  return a.a ⊗ (a.b * b)
end

using LinearAlgebra:
  LinearAlgebra,
  Diagonal,
  Eigen,
  SVD,
  diag,
  eigen,
  eigvals,
  lq,
  mul!,
  norm,
  qr,
  svd,
  svdvals,
  tr
diagonal(a::AbstractVecOrMat) = Diagonal(a)
function diagonal(a::KroneckerArray)
  return Diagonal(a.a) ⊗ Diagonal(a.b)
end

# TODO: Overload `similar` instead?
function LinearAlgebra.matprod_dest(a::KroneckerArray, b::KroneckerArray, elt)
  return LinearAlgebra.matprod_dest(a.a, b.a, elt) ⊗
         LinearAlgebra.matprod_dest(a.b, b.b, elt)
end
function LinearAlgebra.mul!(c::KroneckerArray, a::KroneckerArray, b::KroneckerArray)
  mul!(c.a, a.a, b.a)
  mul!(c.b, a.b, b.b)
  return c
end
function LinearAlgebra.tr(a::KroneckerArray)
  return tr(a.a) ⊗ tr(a.b)
end
function LinearAlgebra.norm(a::KroneckerArray, p::Int=2)
  return norm(a.a, p) ⊗ norm(a.b, p)
end
function LinearAlgebra.diag(a::KroneckerArray)
  return diag(a.a) ⊗ diag(a.b)
end
function LinearAlgebra.svd(a::KroneckerArray)
  Fa = svd(a.a)
  Fb = svd(a.b)
  return SVD(Fa.U ⊗ Fb.U, Fa.S ⊗ Fb.S, Fa.Vt ⊗ Fb.Vt)
end
function LinearAlgebra.svdvals(a::KroneckerArray)
  return svdvals(a.a) ⊗ svdvals(a.b)
end
function LinearAlgebra.eigen(a::KroneckerArray)
  Fa = eigen(a.a)
  Fb = eigen(a.b)
  return Eigen(Fa.values ⊗ Fb.values, Fa.vectors ⊗ Fb.vectors)
end
function LinearAlgebra.eigvals(a::KroneckerArray)
  return eigvals(a.a) ⊗ eigvals(a.b)
end

struct KroneckerQ{A,B}
  a::A
  b::B
end
function Base.:*(a::KroneckerQ, b::KroneckerQ)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.:*(a::KroneckerQ, b::KroneckerMatrix)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.:*(a::KroneckerMatrix, b::KroneckerQ)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.adjoint(a::KroneckerQ)
  return KroneckerQ(a.a', a.b')
end

struct KroneckerQR{QQ,RR}
  Q::QQ
  R::RR
end
Base.iterate(F::KroneckerQR) = (F.Q, Val(:R))
Base.iterate(F::KroneckerQR, ::Val{:R}) = (F.R, Val(:done))
Base.iterate(F::KroneckerQR, ::Val{:done}) = nothing
function ⊗(a::LinearAlgebra.QRCompactWYQ, b::LinearAlgebra.QRCompactWYQ)
  return KroneckerQ(a, b)
end
function LinearAlgebra.qr(a::KroneckerArray)
  Fa = qr(a.a)
  Fb = qr(a.b)
  return KroneckerQR(Fa.Q ⊗ Fb.Q, Fa.R ⊗ Fb.R)
end

struct KroneckerLQ{LL,QQ}
  L::LL
  Q::QQ
end
Base.iterate(F::KroneckerLQ) = (F.L, Val(:Q))
Base.iterate(F::KroneckerLQ, ::Val{:Q}) = (F.Q, Val(:done))
Base.iterate(F::KroneckerLQ, ::Val{:done}) = nothing
function ⊗(a::LinearAlgebra.LQPackedQ, b::LinearAlgebra.LQPackedQ)
  return KroneckerQ(a, b)
end
function LinearAlgebra.lq(a::KroneckerArray)
  Fa = lq(a.a)
  Fb = lq(a.b)
  return KroneckerLQ(Fa.L ⊗ Fb.L, Fa.Q ⊗ Fb.Q)
end

function Base.:-(a::KroneckerArray)
  return (-a.a) ⊗ a.b
end
for op in (:+, :-)
  @eval begin
    function Base.$op(a::KroneckerArray, b::KroneckerArray)
      if a.b == b.b
        return $op(a.a, b.a) ⊗ a.b
      elseif a.a == b.a
        return a.a ⊗ $op(a.b, b.b)
      end
      return throw(
        ArgumentError(
          "KroneckerArray addition is only supported when the first or secord arguments match.",
        ),
      )
    end
  end
end

end
