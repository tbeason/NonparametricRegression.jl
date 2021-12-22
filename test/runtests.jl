using NonparametricRegression
using KernelFunctions, LinearAlgebra
using StableRNGs, Distributions
using Test


rng = StableRNG(123)
vn = randn(rng,30)
vm = rand(rng,20,20)

# write tests here

@testset "NormalKernel" begin 
    nk1 = NormalKernel(1)
    @test nk1.(vn,0) ≈ pdf.(Normal(0,1),vn)

    nk01 = NormalKernel(0.1)
    @test nk01.(vn,0) ≈ pdf.(Normal(0,0.1),vn)

    @test kernelmatrix(nk1,vn,vn) ≈ nk1.(vn,vn')
end


@testset "Utilities" begin

    @test NonparametricRegression.normalizecols(vm) ≈ vm ./ sum(vm,dims=1) 
    
    vm0 = copy(vm)
    NonparametricRegression.zerodiagonal!(vm0)
    @test tr(vm0) ≈ 0
end


@testset "KDE" begin
    @test NonparametricRegression.silvermanbw(vn) ≈ 0.4183944776432713

    kd = NonparametricRegression.densityestimator(randn(rng,2000),[0.0,1.0])

    @test kd ≈ pdf.(Normal(0,1),[0.0,1.0]) rtol=1e-2
end

