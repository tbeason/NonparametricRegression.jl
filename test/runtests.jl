using NonparametricRegression, Distributions
using Test

# write tests here

@testset "NormalKernel" begin 
    nk1 = NormalKernel(1)
    @test nk1.(0:2,0) ≈ pdf.(Normal(0,1),0:2)

    nk01 = NormalKernel(0.1)
    @test nk01.(0:2,0) ≈ pdf.(Normal(0,0.1),0:2)
end
