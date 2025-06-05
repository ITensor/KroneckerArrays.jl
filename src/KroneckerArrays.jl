module KroneckerArrays

using GPUArraysCore: GPUArraysCore

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
  elt = promote_type(eltype(a), eltype(b))
  return KroneckerArray(convert(AbstractArray{elt}, a), convert(AbstractArray{elt}, b))
end
const KroneckerMatrix{T,A<:AbstractMatrix{T},B<:AbstractMatrix{T}} = KroneckerArray{T,2,A,B}
const KroneckerVector{T,A<:AbstractVector{T},B<:AbstractVector{T}} = KroneckerArray{T,1,A,B}

function Base.copy(a::KroneckerArray)
  return copy(a.a) ⊗ copy(a.b)
end
function Base.copyto!(dest::KroneckerArray, src::KroneckerArray)
  copyto!(dest.a, src.a)
  copyto!(dest.b, src.b)
  return dest
end

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
function Base.similar(
  arrayt::Type{<:KroneckerArray{<:Any,<:Any,A,B}},
  axs::Tuple{
    CartesianProductUnitRange{<:Integer},Vararg{CartesianProductUnitRange{<:Integer}}
  },
) where {A,B}
  return similar(A, map(ax -> ax.product.a, axs)) ⊗ similar(B, map(ax -> ax.product.b, axs))
end
function Base.similar(
  ::Type{<:KroneckerArray{<:Any,<:Any,A,B}}, sz::Tuple{Int,Vararg{Int}}
) where {A,B}
  return similar(promote_type(A, B), sz)
end

function flatten(t::Tuple{Tuple,Tuple,Vararg{Tuple}})
  return (t[1]..., flatten(Base.tail(t))...)
end
function flatten(t::Tuple{Tuple})
  return t[1]
end
flatten(::Tuple{}) = ()
function interleave(x::Tuple, y::Tuple)
  length(x) == length(y) || throw(ArgumentError("Tuples must have the same length."))
  xy = ntuple(i -> (x[i], y[i]), length(x))
  return flatten(xy)
end
function kron_nd(a::AbstractArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N}
  a′ = reshape(a, interleave(size(a), ntuple(one, N)))
  b′ = reshape(b, interleave(ntuple(one, N), size(b)))
  c′ = permutedims(a′ .* b′, reverse(ntuple(identity, 2N)))
  sz = ntuple(i -> size(a, i) * size(b, i), N)
  return permutedims(reshape(c′, sz), reverse(ntuple(identity, N)))
end
kron_nd(a::AbstractMatrix, b::AbstractMatrix) = kron(a, b)
kron_nd(a::AbstractVector, b::AbstractVector) = kron(a, b)

Base.collect(a::KroneckerArray) = kron_nd(a.a, a.b)

function Base.Array{T,N}(a::KroneckerArray{S,N}) where {T,S,N}
  return convert(Array{T,N}, collect(a))
end

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
argument_types(a::KroneckerArray) = argument_types(typeof(a))
argument_types(::Type{<:KroneckerArray{<:Any,<:Any,A,B}}) where {A,B} = (A, B)

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

⊗(a::AbstractArray, b::AbstractArray) = KroneckerArray(a, b)
⊗(a::Number, b::Number) = a * b
⊗(a::Number, b::AbstractArray) = a * b
⊗(a::AbstractArray, b::Number) = a * b

function Base.getindex(a::KroneckerArray, i::Integer)
  return a[CartesianIndices(a)[i]]
end

function Base.getindex(a::KroneckerArray{<:Any,N}, I::Vararg{Integer,N}) where {N}
  return error("Not implemented.")
end

function Base.getindex(a::KroneckerMatrix, i1::Integer, i2::Integer)
  GPUArraysCore.assertscalar("getindex")
  # Code logic from Kronecker.jl:
  # https://github.com/MichielStock/Kronecker.jl/blob/v0.5.5/src/base.jl#L101-L105
  k, l = size(a.b)
  return a.a[cld(i1, k), cld(i2, l)] * a.b[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

function Base.getindex(a::KroneckerVector, i::Integer)
  GPUArraysCore.assertscalar("getindex")
  k = length(a.b)
  return a.a[cld(i, k)] * a.b[(i - 1) % k + 1]
end

## function Base.getindex(a::KroneckerVector, i::CartesianProduct)
##   return a.a[i.a] ⊗ a.b[i.b]
## end
function Base.getindex(a::KroneckerArray{<:Any,N}, I::Vararg{CartesianProduct,N}) where {N}
  return a.a[map(Base.Fix2(getfield, :a), I)...] ⊗ a.b[map(Base.Fix2(getfield, :b), I)...]
end
# Fix ambigiuity error.
Base.getindex(a::KroneckerArray{<:Any,0}) = a.a[] * a.b[]

function Base.:(==)(a::KroneckerArray, b::KroneckerArray)
  return a.a == b.a && a.b == b.b
end
function Base.isapprox(a::KroneckerArray, b::KroneckerArray; kwargs...)
  return isapprox(a.a, b.a; kwargs...) && isapprox(a.b, b.b; kwargs...)
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

function Base.map!(::typeof(identity), dest::KroneckerArray, a::KroneckerArray)
  dest.a .= a.a
  dest.b .= a.b
  return dest
end
function Base.map!(::typeof(+), dest::KroneckerArray, a::KroneckerArray, b::KroneckerArray)
  if a.b == b.b
    map!(+, dest.a, a.a, b.a)
    dest.b .= a.b
  elseif a.a == b.a
    dest.a .= a.a
    map!(+, dest.b, a.b, b.b)
  else
    throw(
      ArgumentError(
        "KroneckerArray addition is only supported when the first or second arguments match.",
      ),
    )
  end
  return dest
end
function Base.map!(
  f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerArray, a::KroneckerArray
)
  dest.a .= f.f.(f.x, a.a)
  dest.b .= a.b
  return dest
end
function Base.map!(
  f::Base.Fix2{typeof(*),<:Number}, dest::KroneckerArray, a::KroneckerArray
)
  dest.a .= a.a
  dest.b .= f.f.(a.b, f.x)
  return dest
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
diagonal(a::AbstractArray) = Diagonal(a)
function diagonal(a::KroneckerArray)
  return Diagonal(a.a) ⊗ Diagonal(a.b)
end

function Base.:*(a::KroneckerArray, b::KroneckerArray)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function LinearAlgebra.mul!(
  c::KroneckerArray, a::KroneckerArray, b::KroneckerArray, α::Number, β::Number
)
  iszero(β) ||
    iszero(c) ||
    throw(
      ArgumentError(
        "Can't multiple KroneckerArrays with nonzero β and nonzero destination."
      ),
    )
  mul!(c.a, a.a, b.a)
  mul!(c.b, a.b, b.b, α, β)
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
function Base.:*(a::KroneckerQ, b::KroneckerArray)
  return (a.a * b.a) ⊗ (a.b * b.b)
end
function Base.:*(a::KroneckerArray, b::KroneckerQ)
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

using DerivableInterfaces: DerivableInterfaces, zero!
function DerivableInterfaces.zero!(a::KroneckerArray)
  zero!(a.a)
  zero!(a.b)
  return a
end

using FillArrays: Eye
const EyeKronecker{T,A<:Eye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerEye{T,A<:AbstractMatrix{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}
const EyeEye{T,A<:Eye{T},B<:Eye{T}} = KroneckerMatrix{T,A,B}

function Base.:*(a::Number, b::EyeKronecker)
  return b.a ⊗ (a * b.b)
end
function Base.:*(a::Number, b::KroneckerEye)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::Number, b::EyeEye)
  return (a * b.a) ⊗ b.b
end
function Base.:*(a::EyeKronecker, b::Number)
  return a.a ⊗ (a.b * b)
end
function Base.:*(a::KroneckerEye, b::Number)
  return (a.a * b) ⊗ a.b
end
function Base.:*(a::EyeEye, b::Number)
  return a.a ⊗ (a.b * b)
end

function Base.:-(a::EyeKronecker)
  return a.a ⊗ (-a.b)
end
function Base.:-(a::KroneckerEye)
  return (-a.a) ⊗ a.b
end
function Base.:-(a::EyeEye)
  return (-a.a) ⊗ a.b
end
for op in (:+, :-)
  @eval begin
    function Base.$op(a::EyeKronecker, b::EyeKronecker)
      if a.a ≠ b.a
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return a.a ⊗ $op(a.b, b.b)
    end
    function Base.$op(a::KroneckerEye, b::KroneckerEye)
      if a.b ≠ b.b
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return $op(a.a, b.a) ⊗ a.b
    end
    function Base.$op(a::EyeEye, b::EyeEye)
      if a.b ≠ b.b
        return throw(
          ArgumentError(
            "KroneckerArray addition is only supported when the first or secord arguments match.",
          ),
        )
      end
      return $op(a.a, b.a) ⊗ a.b
    end
  end
end

function Base.map!(::typeof(identity), dest::EyeKronecker, a::EyeKronecker)
  dest.b .= a.b
  return dest
end
function Base.map!(::typeof(identity), dest::KroneckerEye, a::KroneckerEye)
  dest.a .= a.a
  return dest
end
function Base.map!(::typeof(identity), dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::typeof(+), dest::EyeKronecker, a::EyeKronecker, b::EyeKronecker)
  if dest.a ≠ a.a ≠ b.a
    throw(
      ArgumentError(
        "KroneckerArray addition is only supported when the first or second arguments match.",
      ),
    )
  end
  map!(f, dest.b, a.b, b.b)
  return dest
end
function Base.map!(f::typeof(+), dest::KroneckerEye, a::KroneckerEye, b::KroneckerEye)
  if dest.b ≠ a.b ≠ b.b
    throw(
      ArgumentError(
        "KroneckerArray addition is only supported when the first or second arguments match.",
      ),
    )
  end
  map!(f, dest.a, a.a, b.a)
  return dest
end
function Base.map!(f::typeof(+), dest::EyeEye, a::EyeEye, b::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::EyeKronecker, a::EyeKronecker)
  dest.b .= f.f.(f.x, a.b)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  dest.a .= f.f.(f.x, a.a)
  return dest
end
function Base.map!(f::Base.Fix1{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeKronecker, a::EyeKronecker)
  dest.b .= f.f.(a.b, f.x)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::KroneckerEye, a::KroneckerEye)
  dest.a .= f.f.(a.a, f.x)
  return dest
end
function Base.map!(f::Base.Fix2{typeof(*),<:Number}, dest::EyeEye, a::EyeEye)
  return error("Can't write in-place.")
end

using MatrixAlgebraKit:
  MatrixAlgebraKit,
  AbstractAlgorithm,
  TruncationStrategy,
  default_eig_algorithm,
  default_eigh_algorithm,
  default_lq_algorithm,
  default_polar_algorithm,
  default_qr_algorithm,
  default_svd_algorithm,
  eig_full!,
  eig_trunc!,
  eig_vals!,
  eigh_full!,
  eigh_trunc!,
  eigh_vals!,
  initialize_output,
  left_null!,
  left_orth!,
  left_polar!,
  lq_compact!,
  lq_full!,
  qr_compact!,
  qr_full!,
  right_null!,
  right_orth!,
  right_polar!,
  svd_compact!,
  svd_full!,
  svd_trunc!,
  svd_vals!,
  truncate!

struct KroneckerAlgorithm{A,B} <: AbstractAlgorithm
  a::A
  b::B
end

using MatrixAlgebraKit:
  copy_input,
  eig_full,
  eigh_full,
  qr_compact,
  qr_full,
  left_polar,
  lq_compact,
  lq_full,
  right_polar,
  svd_compact,
  svd_full

for f in [
  :eig_full,
  :eigh_full,
  :qr_compact,
  :qr_full,
  :left_polar,
  :lq_compact,
  :lq_full,
  :right_polar,
  :svd_compact,
  :svd_full,
]
  @eval begin
    function MatrixAlgebraKit.copy_input(::typeof($f), a::KroneckerMatrix)
      return copy_input($f, a.a) ⊗ copy_input($f, a.b)
    end
  end
end

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  @eval begin
    function MatrixAlgebraKit.$f(
      A::Type{<:KroneckerMatrix}; kwargs1=(;), kwargs2=(;), kwargs...
    )
      A1, A2 = argument_types(A)
      return KroneckerAlgorithm(
        $f(A1; kwargs..., kwargs1...), $f(A2; kwargs..., kwargs2...)
      )
    end
  end
end

# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(qr_compact!), A::Type{<:KroneckerMatrix}; kwargs...
)
  return default_qr_algorithm(A; kwargs...)
end
# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/32 is merged.
function MatrixAlgebraKit.default_algorithm(
  ::typeof(qr_full!), A::Type{<:KroneckerMatrix}; kwargs...
)
  return default_qr_algorithm(A; kwargs...)
end

for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), a::KroneckerMatrix, alg::KroneckerAlgorithm
    )
      return initialize_output($f, a.a, alg.a) .⊗ initialize_output($f, a.b, alg.b)
    end
    function MatrixAlgebraKit.$f(
      a::KroneckerMatrix, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
    )
      $f(a.a, Base.Fix2(getfield, :a).(F), alg.a; kwargs..., kwargs1...)
      $f(a.b, Base.Fix2(getfield, :b).(F), alg.b; kwargs..., kwargs2...)
      return F
    end
  end
end

for f in [:eig_vals!, :eigh_vals!, :svd_vals!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(
      ::typeof($f), a::KroneckerMatrix, alg::KroneckerAlgorithm
    )
      return initialize_output($f, a.a, alg.a) ⊗ initialize_output($f, a.b, alg.b)
    end
    function MatrixAlgebraKit.$f(a::KroneckerMatrix, F, alg::KroneckerAlgorithm)
      $f(a.a, F.a, alg.a)
      $f(a.b, F.b, alg.b)
      return F
    end
  end
end

for f in [:eig_trunc!, :eigh_trunc!, :svd_trunc!]
  @eval begin
    function MatrixAlgebraKit.truncate!(
      ::typeof($f),
      (D, V)::Tuple{KroneckerMatrix,KroneckerMatrix},
      strategy::TruncationStrategy,
    )
      return throw(MethodError(truncate!, ($f, (D, V), strategy)))
    end
  end
end

for f in [:left_orth!, :right_orth!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
      return initialize_output($f, a.a) .⊗ initialize_output($f, a.b)
    end
  end
end

for f in [:left_null!, :right_null!]
  @eval begin
    function MatrixAlgebraKit.initialize_output(::typeof($f), a::KroneckerMatrix)
      return initialize_output($f, a.a) ⊗ initialize_output($f, a.b)
    end
    function MatrixAlgebraKit.$f(a::KroneckerMatrix, F; kwargs1=(;), kwargs2=(;), kwargs...)
      $f(a.a, F.a; kwargs..., kwargs1...)
      $f(a.b, F.b; kwargs..., kwargs2...)
      return F
    end
  end
end

####################################################################################
# Special cases for MatrixAlgebraKit factorizations of `Eye(n) ⊗ A` and
# `A ⊗ Eye(n)` where `A`.
# TODO: Delete this once https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/34
# is merged.

using FillArrays: SquareEye
const SquareEyeKronecker{T,A<:SquareEye{T},B<:AbstractMatrix{T}} = KroneckerMatrix{T,A,B}
const KroneckerSquareEye{T,A<:AbstractMatrix{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}
const SquareEyeSquareEye{T,A<:SquareEye{T},B<:SquareEye{T}} = KroneckerMatrix{T,A,B}

struct SquareEyeAlgorithm <: AbstractAlgorithm end

# Defined to avoid type piracy.
_copy_input_squareeye(f::F, a) where {F} = copy_input(f, a)
_copy_input_squareeye(f::F, a::SquareEye) where {F} = a

for f in [
  :eig_full,
  :eigh_full,
  :qr_compact,
  :qr_full,
  :left_polar,
  :lq_compact,
  :lq_full,
  :right_polar,
  :svd_compact,
  :svd_full,
]
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.copy_input(::typeof($f), a::$T)
        return _copy_input_squareeye($f, a.a) ⊗ _copy_input_squareeye($f, a.b)
      end
    end
  end
end

for f in [
  :default_eig_algorithm,
  :default_eigh_algorithm,
  :default_lq_algorithm,
  :default_qr_algorithm,
  :default_polar_algorithm,
  :default_svd_algorithm,
]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a) = $f(a)
    $f′(a::Type{<:SquareEye}) = SquareEyeAlgorithm()
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.$f(A::Type{<:$T}; kwargs1=(;), kwargs2=(;), kwargs...)
        A1, A2 = argument_types(A)
        return KroneckerAlgorithm(
          $f′(A1; kwargs..., kwargs1...), $f′(A2; kwargs..., kwargs2...)
        )
      end
    end
  end
end

# Defined to avoid type piracy.
_initialize_output_squareeye(f::F, a, alg) where {F} = initialize_output(f, a, alg)
for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_polar!,
]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, a)
  end
end
for f in [:svd_compact!, :svd_full!]
  @eval begin
    _initialize_output_squareeye(::typeof($f), a::SquareEye, alg) = (a, a, a)
  end
end

for f in [
  :eig_full!,
  :eigh_full!,
  :qr_compact!,
  :qr_full!,
  :left_polar!,
  :lq_compact!,
  :lq_full!,
  :right_polar!,
  :svd_compact!,
  :svd_full!,
]
  f′ = Symbol("_", f, "_squareeye")
  @eval begin
    $f′(a, F, alg; kwargs...) = $f(a, F, alg; kwargs...)
    $f′(a, F, alg::SquareEyeAlgorithm) = F
  end
  for T in [:SquareEyeKronecker, :KroneckerSquareEye, :SquareEyeSquareEye]
    @eval begin
      function MatrixAlgebraKit.initialize_output(
        ::typeof($f), a::$T, alg::KroneckerAlgorithm
      )
        return _initialize_output_squareeye($f, a.a, alg.a) .⊗
               _initialize_output_squareeye($f, a.b, alg.b)
      end
      function MatrixAlgebraKit.$f(
        a::$T, F, alg::KroneckerAlgorithm; kwargs1=(;), kwargs2=(;), kwargs...
      )
        $f′(a.a, Base.Fix2(getfield, :a).(F), alg.a; kwargs..., kwargs1...)
        $f′(a.b, Base.Fix2(getfield, :b).(F), alg.b; kwargs..., kwargs2...)
        return F
      end
    end
  end
end

end
