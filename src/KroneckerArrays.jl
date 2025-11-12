module KroneckerArrays

export kroneckerfactors, kroneckerfactortypes
export times, ×, cartesianproduct, cartesianrange, unproduct
export ⊗, ×

# Imports
# -------
import Base.Broadcast as BC
using LinearAlgebra: LinearAlgebra, Diagonal, diag, isdiag
using DiagonalArrays: DiagonalArrays
using DerivableInterfaces: DerivableInterfaces
using MapBroadcast: MapBroadcast, MapFunction, LinearCombination, Summed
using GPUArraysCore: GPUArraysCore
using Adapt: Adapt

# Interfaces
# ----------
@doc """
    kroneckerfactors(x) -> Tuple
    kroneckerfactors(x, i) = kroneckerfactors(x)[i]

Extract the factors of `x`, where `x` is an object that represents a lazily composed product type.
""" kroneckerfactors
# note: this is `Int` instead of `Integer` to avoid ambiguities downstream
@inline kroneckerfactors(x, i::Int) = kroneckerfactors(x)[i]

@doc """
    kroneckerfactortypes(x) -> Tuple
    kroneckerfactortypes(x, i) = kroneckerfactortypes(x)[i]

Extract the types of the factors of `x`, where `x` is an object or type that represents a lazily composed product type.
""" kroneckerfactortypes
# note: this is `Int` instead of `Integer` to avoid ambiguities downstream
@inline kroneckerfactortypes(x, i::Int) = kroneckerfactortypes(x)[i]
kroneckerfactortypes(x) = kroneckerfactortypes(typeof(x))
kroneckerfactortypes(T::Type) = throw(MethodError(kroneckerfactortypes, (T,)))

@doc """
    ⊗(args...)
    otimes(args...)

Construct an object that represents the Kronecker product of the provided `args`.
""" (⊗)
function ⊗(a, b) end
const otimes = ⊗ # non-unicode alternative

# Includes
# --------
include("cartesianproduct.jl")
include("kroneckerarray.jl")
include("linearalgebra.jl")
include("matrixalgebrakit.jl")
include("fillarrays.jl")

end
