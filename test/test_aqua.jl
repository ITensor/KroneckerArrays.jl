using Aqua: Aqua
using KroneckerArrays: KroneckerArrays
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(KroneckerArrays)
end
