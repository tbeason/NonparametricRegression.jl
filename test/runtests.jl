using NonparametricRegression
using LinearAlgebra
using StableRNGs, Distributions
using Test


rng = StableRNG(123)
vn = randn(rng,30)
vu = rand(rng,20)

# write tests here
@testset verbose=true "NonparametricRegression Tests" begin

N = 2500
xs = randn(rng,N)
ys = 2 .+ xs.^2 .+ 0.1 .* randn(rng,N)
xg = collect(-1:0.5:1)


@testset "Exported" begin
    @testset "Kernels" begin
        @testset "NormalKernel" begin 
            nk1 = NormalKernel(1)
            @test nk1.(vn,0) ≈ pdf.(Normal(0,1),vn)
        
            nk01 = NormalKernel(0.1)
            @test nk01.(vn,0) ≈ pdf.(Normal(0,0.1),vn)
    
            @test bandwidth(nk1) == 1
    
            @test NonparametricRegression.scaling(nk1) ≈ 1/sqrt(2*pi)
        end
    
        @testset "UniformKernel" begin 
            u1 = UniformKernel(1)
            @test u1.(vu,0) ≈ pdf.(Uniform(-1,1),vu)
        
            u01 = UniformKernel(0.1)
            @test u01.(vu,0) ≈ pdf.(Uniform(-0.1,0.1),vu)
    
            @test bandwidth(u1) == 1
    
            @test NonparametricRegression.scaling(u1) ≈ 1/2
        end
    
    
        @testset "EpanechnikovKernel" begin 
            ek1 = EpanechnikovKernel(1)
            @test ek1.(vu,0) ≈ 3/4 .* (1 .- vu.^2)
        
            ek01 = EpanechnikovKernel(0.1)
            @test ek01.(vu,0) ≈ 3/4 .* (1 .- (vu./0.1).^2) ./ 0.1 .* (abs.(vu./0.1) .<= 1)
    
            @test bandwidth(ek1) == 1
    
            @test NonparametricRegression.scaling(ek1) ≈ 3/4
        end
    end





    @testset "localconstant" begin
        lc1 = localconstant(xs,ys,xg,1.0)
        @test lc1[3] ≈ 2 rtol=0.5

        loocv_h = leaveoneoutCV(xs,ys; method=:lc)
        # println(loocv_h)
        lc2 = localconstant(xs,ys,xg,loocv_h)
        @test lc2[3] ≈ 2 rtol=1e-2

        aicc_h = optimizeAICc(xs,ys; method=:lc)
        # println(aicc_h)
        lc3 = localconstant(xs,ys,xg,aicc_h)
        @test lc3[3] ≈ 2 rtol=1e-2

        # println([lc2 ;; lc3])

        nplc1 = npregress(xs,ys,xg,1.0; method=:lc)
        @test nplc1 ≈ lc1

        nplc2 = npregress(xs,ys,xg; method=:lc, bandwidthselection=:loocv)
        @test nplc2 ≈ lc2

        nplc3 = npregress(xs,ys,xg; method=:lc, bandwidthselection=:aicc)
        @test nplc3 ≈ lc3


    end

    
    @testset "locallinear" begin
        ll1 = locallinear(xs,ys,xg,1.0)
        @test ll1[3] ≈ 2 rtol=0.5

        loocv_h = leaveoneoutCV(xs,ys; method=:ll)
        # println(loocv_h)
        ll2 = locallinear(xs,ys,xg,loocv_h)
        @test ll2[3] ≈ 2 rtol=1e-2

        aicc_h = optimizeAICc(xs,ys; method=:ll)
        # println(aicc_h)
        ll3 = locallinear(xs,ys,xg,aicc_h)
        @test ll3[3] ≈ 2 rtol=1e-2

        # println([ll2 ;; ll3])

        npll1 = npregress(xs,ys,xg,1.0; method=:ll)
        @test npll1 ≈ ll1

        npll2 = npregress(xs,ys,xg; method=:ll, bandwidthselection=:loocv)
        @test npll2 ≈ ll2

        npll3 = npregress(xs,ys,xg; method=:ll, bandwidthselection=:aicc)
        @test npll3 ≈ ll3

        ll3ab = llalphabeta(xs,ys,xg,aicc_h)
        @test first(ll3ab) ≈ ll3
    end

    @testset "errors" begin
        @test_throws ErrorException npregress(xs,ys,xg,1; method=:hello)
        @test_throws ErrorException npregress(xs,ys,xg; method=:lc, bandwidthselection=:hello)
        @test_throws ErrorException npregress(xs,ys,xg; method=:hello, bandwidthselection=:loocv)
        @test_throws ErrorException npregress(xs,ys,xg; method=:hello, bandwidthselection=:aicc)
    end

end


# Include varying coefficient tests
include("test_varyingcoefficient.jl")


@testset "Unexported" begin
    @testset "Utilities" begin
        @test NonparametricRegression.AICc(1,1,100) ≈ 0 + 1 + 2*(1+1)/(100-1-2)
    end
    
    
    @testset "KDE" begin
        @test NonparametricRegression.silvermanbw(vn) ≈ 0.4183944776432713
    
        kd = NonparametricRegression.densityestimator(xs,[0.0,1.0])
    
        @test kd ≈ pdf.(Normal(0,1),[0.0,1.0]) atol=1e-2
    end
end


end # end all tests

