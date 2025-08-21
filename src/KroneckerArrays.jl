module KroneckerArrays

export ⊗, ×

include("cartesianproduct.jl")
include("kroneckerarray.jl")
include("linearalgebra.jl")
include("matrixalgebrakit.jl")
include("fillarrays/kroneckerarray.jl")
include("fillarrays/linearalgebra.jl")
# include("fillarrays/matrixalgebrakit.jl")
# include("fillarrays/matrixalgebrakit_truncate.jl")

end
