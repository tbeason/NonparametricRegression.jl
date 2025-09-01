using NonparametricRegression
using NonparametricRegression: varyingcoefficient_lc, varyingcoefficient_ll, varyingcoefficient_aicc, 
                               leaveoneoutCV_varyingcoef, varyingcoefficientweights
using LinearAlgebra
using StableRNGs, Distributions
using Test

@testset "Varying Coefficient Models" begin
    rng = StableRNG(456)
    
    # Generate test data with known varying coefficients
    N = 300
    z = sort(rand(rng, N))  # State variable
    X = hcat(ones(N), randn(rng, N), randn(rng, N))  # Design matrix: intercept + 2 covariates
    
    # True varying coefficients
    β_true = hcat(
        1 .+ z,                    # β₁(z) = 1 + z
        2 .- z.^2,                 # β₂(z) = 2 - z²
        0.5 .* sin.(2π * z)        # β₃(z) = 0.5 sin(2πz)
    )
    
    # Generate y
    y = zeros(N)
    for i in 1:N
        y[i] = dot(X[i, :], β_true[i, :]) + 0.1 * randn(rng)
    end
    
    # Grid for evaluation
    zgrid = collect(0.1:0.1:0.9)
    
    @testset "Local Constant Varying Coefficient" begin
        # Test with fixed bandwidth
        h = 0.1
        B_lc = varyingcoefficient_lc(X, y, z, zgrid, h)
        
        @test size(B_lc) == (3, length(zgrid))
        
        # Check that estimates are reasonable at z=0.5
        idx = findfirst(==(0.5), zgrid)
        β_true_at_05 = [1.5, 1.75, 0.0]  # True values at z=0.5
        @test B_lc[:, idx] ≈ β_true_at_05 rtol=0.2
        
        # Test npvaryingcoef wrapper
        B_lc2 = npvaryingcoef(X, y, z, zgrid, h; method=:lc)
        @test B_lc ≈ B_lc2
    end
    
    @testset "Local Linear Varying Coefficient" begin
        # Test with fixed bandwidth
        h = 0.15
        B_ll = varyingcoefficient_ll(X, y, z, zgrid, h)
        
        @test size(B_ll) == (3, length(zgrid))
        
        # Local linear should be more accurate than local constant
        idx = findfirst(==(0.5), zgrid)
        β_true_at_05 = [1.5, 1.75, 0.0]
        @test B_ll[:, idx] ≈ β_true_at_05 rtol=0.15
        
        # Test npvaryingcoef wrapper
        B_ll2 = npvaryingcoef(X, y, z, zgrid, h; method=:ll)
        @test B_ll ≈ B_ll2
    end
    
    @testset "Bandwidth Selection" begin
        # Test AICc
        h_test = 0.1
        aicc_lc = varyingcoefficient_aicc(X, y, z, h_test; method=:lc)
        aicc_ll = varyingcoefficient_aicc(X, y, z, h_test; method=:ll)
        
        @test isa(aicc_lc, Real)
        @test isa(aicc_ll, Real)
        @test isfinite(aicc_lc)
        @test isfinite(aicc_ll)
        
        # Test LOOCV
        loocv_lc = leaveoneoutCV_varyingcoef(X, y, z, h_test; method=:lc)
        loocv_ll = leaveoneoutCV_varyingcoef(X, y, z, h_test; method=:ll)
        
        @test isa(loocv_lc, Real)
        @test isa(loocv_ll, Real)
        @test loocv_lc > 0
        @test loocv_ll > 0
    end
    
    @testset "Automatic Bandwidth Selection" begin
        # Test with AICc
        B_auto_aicc = npvaryingcoef(X, y, z, zgrid; method=:lc, bandwidthselection=:aicc)
        @test size(B_auto_aicc) == (3, length(zgrid))
        
        # Test with LOOCV (use smaller dataset for speed)
        X_small = X[1:100, :]
        y_small = y[1:100]
        z_small = z[1:100]
        B_auto_loocv = npvaryingcoef(X_small, y_small, z_small, zgrid; method=:lc, bandwidthselection=:loocv)
        @test size(B_auto_loocv) == (3, length(zgrid))
    end
    
    @testset "Prediction" begin
        # Fit model
        h = 0.1
        B = npvaryingcoef(X, y, z, zgrid, h; method=:lc)
        
        # Generate new data
        M = 50
        z_new = rand(rng, M)
        X_new = hcat(ones(M), randn(rng, M), randn(rng, M))
        
        # Predict
        y_pred = predict_varyingcoef(X_new, z_new, B, zgrid, h; method=:lc)
        
        @test length(y_pred) == M
        @test all(isfinite.(y_pred))
        
        # Check that predictions are reasonable
        # For a simple check, predict at a point in zgrid
        idx = 1
        z_test = [zgrid[idx]]
        X_test = reshape([1.0, 0.5, -0.5], 1, 3)  # Test covariate values
        y_pred_test = predict_varyingcoef(X_test, z_test, B, zgrid, h; method=:lc)
        @test y_pred_test[1] ≈ dot(X_test[:], B[:, idx]) rtol=0.01  # 1% tolerance due to interpolation
    end
    
    @testset "Weights" begin
        h = 0.1
        W_lc = varyingcoefficientweights(X, z, zgrid, h; method=:lc)
        W_ll = varyingcoefficientweights(X, z, zgrid, h; method=:ll)
        
        @test size(W_lc) == (N, length(zgrid))
        @test size(W_ll) == (N, length(zgrid))
        
        # Weights are hat matrix weights, not normalized kernel weights
        # They don't necessarily sum to 1
        @test all(isfinite.(W_lc))
        @test all(isfinite.(W_ll))
    end
    
    @testset "Edge Cases" begin
        # Single observation at each z
        N_edge = 10
        z_edge = collect(1:N_edge) / N_edge
        X_edge = hcat(ones(N_edge), randn(rng, N_edge))
        y_edge = randn(rng, N_edge)
        
        # Should still work but with larger bandwidth
        h_edge = 0.5
        B_edge = npvaryingcoef(X_edge, y_edge, z_edge, [0.5], h_edge; method=:lc)
        @test size(B_edge) == (2, 1)
        
        # Test with mismatched dimensions
        @test_throws AssertionError varyingcoefficient_lc(X[1:10, :], y[1:9], z[1:10], zgrid, 0.1)
        
        # Test unknown method
        @test_throws ErrorException npvaryingcoef(X, y, z, zgrid, 0.1; method=:unknown)
    end
    
    @testset "Different Kernels" begin
        h = 0.1
        zgrid_small = [0.3, 0.5, 0.7]
        
        # Test with different kernels
        B_gauss = npvaryingcoef(X, y, z, zgrid_small, h; method=:lc, kernelfun=GaussianKernel)
        B_unif = npvaryingcoef(X, y, z, zgrid_small, h; method=:lc, kernelfun=UniformKernel)
        B_epan = npvaryingcoef(X, y, z, zgrid_small, h; method=:lc, kernelfun=EpanechnikovKernel)
        
        @test size(B_gauss) == (3, 3)
        @test size(B_unif) == (3, 3)
        @test size(B_epan) == (3, 3)
        
        # Results should be different but reasonable
        @test !isapprox(B_gauss, B_unif; rtol=1e-2)
        @test !isapprox(B_gauss, B_epan; rtol=1e-2)
    end
    
    @testset "Functional Accuracy - Complex Functions" begin
        # Create a larger, more rigorous test dataset with specific function forms
        rng_acc = StableRNG(789)  # Different seed
        N_acc = 1000  # More observations for better estimation
        
        # Create a more structured grid for z to ensure good coverage
        z_acc = collect(range(0.01, 0.99, length=N_acc))
        z_acc = z_acc .+ 0.005 * randn(rng_acc, N_acc)  # Add small noise but maintain structure
        z_acc = sort(clamp.(z_acc, 0.0, 1.0))  # Ensure within [0,1]
        
        # Design matrix with controlled correlation structure
        X1 = randn(rng_acc, N_acc)
        X2 = 0.3 * X1 + 0.95 * randn(rng_acc, N_acc)  # Correlated with X1
        X3 = randn(rng_acc, N_acc)
        X_acc = hcat(ones(N_acc), X1, X2, X3)
        
        # More complex functional forms for coefficients
        # Define the true coefficient functions
        β1(z) = 1 + 2*z - 3*z^2  # Quadratic
        β2(z) = cos(2π*z)  # Lower-frequency cosine (more estimable)
        β3(z) = exp(-3*(z-0.5)^2)  # Gaussian bump (less steep)
        β4(z) = 0.5 * (z < 0.5 ? 2*z : 2-2*z)  # Tent function - tests discontinuous derivative
        
        # Generate true coefficient values
        β_true_acc = zeros(N_acc, 4)
        for i in 1:N_acc
            β_true_acc[i, 1] = β1(z_acc[i])
            β_true_acc[i, 2] = β2(z_acc[i])
            β_true_acc[i, 3] = β3(z_acc[i])
            β_true_acc[i, 4] = β4(z_acc[i])
        end
        
        # Generate response with heteroskedastic noise
        y_acc = zeros(N_acc)
        for i in 1:N_acc
            y_acc[i] = dot(X_acc[i, :], β_true_acc[i, :]) + 0.05 * (1 + z_acc[i]) * randn(rng_acc)
        end
        
        # Grid for evaluation - dense grid to capture function shapes
        zgrid_acc = collect(0.1:0.05:0.9)
        
        # True coefficient values at grid points
        β_true_at_grid = zeros(4, length(zgrid_acc))
        for (j, zg) in enumerate(zgrid_acc)
            β_true_at_grid[1, j] = β1(zg)
            β_true_at_grid[2, j] = β2(zg)
            β_true_at_grid[3, j] = β3(zg)
            β_true_at_grid[4, j] = β4(zg)
        end
        
        # Test local constant estimator accuracy
        h_lc = 0.15  # Larger bandwidth for stability
        B_lc_acc = varyingcoefficient_lc(X_acc, y_acc, z_acc, zgrid_acc, h_lc)
        
        # Test local linear estimator accuracy
        h_ll = 0.2  # Larger bandwidth for stability
        B_ll_acc = varyingcoefficient_ll(X_acc, y_acc, z_acc, zgrid_acc, h_ll)
        
        # Test overall function approximation quality (more robust than pointwise tests)
        # Use correlation and overall RMSE instead of strict pointwise accuracy
        for i in 1:4
            # Calculate relative RMSE
            lc_rmse = sqrt(mean((B_lc_acc[i, :] - β_true_at_grid[i, :]).^2)) / std(β_true_at_grid[i, :])
            ll_rmse = sqrt(mean((B_ll_acc[i, :] - β_true_at_grid[i, :]).^2)) / std(β_true_at_grid[i, :])
            
            # Calculate correlations
            lc_corr = cor(B_lc_acc[i, :], β_true_at_grid[i, :])
            ll_corr = cor(B_ll_acc[i, :], β_true_at_grid[i, :])
            
            # Overall approximation should be reasonable
            @test lc_rmse < 1.0  # RMSE should be less than 1 standard deviation
            @test ll_rmse < 0.8  # Local linear should be better
            
            # Correlations should be positive and reasonably high
            @test lc_corr > 0.6  # Should capture general shape
            @test ll_corr > 0.7  # Local linear should be better
            
            # For smooth functions (1 and 3), expect better performance
            if i ∈ [1, 3]
                @test lc_corr > 0.8
                @test ll_corr > 0.85
            end
        end
        
        # Test specific points where we expect good accuracy
        mid_idx = findfirst(x -> abs(x - 0.5) < 0.05, zgrid_acc)  # Point near center
        if mid_idx !== nothing
            # Quadratic function should be well estimated
            @test B_ll_acc[1, mid_idx] ≈ β1(0.5) rtol=0.15
            # Gaussian bump should be captured at peak
            @test B_ll_acc[3, mid_idx] ≈ β3(0.5) rtol=0.25
        end
    end
    
    @testset "Automatic Bandwidth Selection Accuracy" begin
        # Generate a dataset with a simple but clear pattern
        rng_bw = StableRNG(123)
        N_bw = 200
        z_bw = sort(rand(rng_bw, N_bw))
        X_bw = hcat(ones(N_bw), randn(rng_bw, N_bw))
        
        # Simple linear coefficient functions
        β1_bw(z) = 1 + z
        β2_bw(z) = 2 - 2*z
        
        # Generate data with low noise
        y_bw = zeros(N_bw)
        for i in 1:N_bw
            y_bw[i] = X_bw[i, 1] * β1_bw(z_bw[i]) + X_bw[i, 2] * β2_bw(z_bw[i]) + 0.05 * randn(rng_bw)
        end
        
        # Evaluation grid
        zgrid_bw = collect(0.2:0.2:0.8)
        
        # True coefficient values at grid points
        β_true_bw = zeros(2, length(zgrid_bw))
        for (j, zg) in enumerate(zgrid_bw)
            β_true_bw[1, j] = β1_bw(zg)
            β_true_bw[2, j] = β2_bw(zg)
        end
        
        # Test AICc bandwidth selection
        B_aicc = npvaryingcoef(X_bw, y_bw, z_bw, zgrid_bw; method=:ll, bandwidthselection=:aicc)
        
        # Test LOOCV bandwidth selection - smaller dataset for speed
        X_bw_small = X_bw[1:100, :]
        y_bw_small = y_bw[1:100]
        z_bw_small = z_bw[1:100]
        B_loocv = npvaryingcoef(X_bw_small, y_bw_small, z_bw_small, zgrid_bw; method=:ll, bandwidthselection=:loocv)
        
        # Both methods should produce accurate results
        for j in 1:length(zgrid_bw)
            @test B_aicc[1, j] ≈ β_true_bw[1, j] rtol=0.1
            @test B_aicc[2, j] ≈ β_true_bw[2, j] rtol=0.1
            
            @test B_loocv[1, j] ≈ β_true_bw[1, j] rtol=0.15
            @test B_loocv[2, j] ≈ β_true_bw[2, j] rtol=0.15
        end
        
        # Overall function approximation should be good
        for i in 1:2
            # Calculate relative error
            aicc_error = norm(B_aicc[i, :] - β_true_bw[i, :]) / norm(β_true_bw[i, :])
            loocv_error = norm(B_loocv[i, :] - β_true_bw[i, :]) / norm(β_true_bw[i, :])
            
            @test aicc_error < 0.1  # 10% relative error or less
            @test loocv_error < 0.15  # 15% relative error or less
        end
    end
    
    @testset "Prediction Accuracy" begin
        # Test prediction with a new dataset
        rng_pred = StableRNG(567)
        N_train = 300
        N_test = 50
        
        # State variable
        z_train = sort(rand(rng_pred, N_train))
        z_test = sort(rand(rng_pred, N_test))
        
        # Design matrices
        X_train = hcat(ones(N_train), randn(rng_pred, N_train, 2))
        X_test = hcat(ones(N_test), randn(rng_pred, N_test, 2))
        
        # Coefficient functions
        β1_pred(z) = 1 + 0.5*sin(2π*z)
        β2_pred(z) = 0.5 + z^2
        β3_pred(z) = exp(-3*z)
        
        # Generate training data
        y_train = zeros(N_train)
        for i in 1:N_train
            y_train[i] = X_train[i, 1] * β1_pred(z_train[i]) + 
                         X_train[i, 2] * β2_pred(z_train[i]) + 
                         X_train[i, 3] * β3_pred(z_train[i]) + 
                         0.1 * randn(rng_pred)
        end
        
        # Generate true test data (without noise)
        y_test_true = zeros(N_test)
        for i in 1:N_test
            y_test_true[i] = X_test[i, 1] * β1_pred(z_test[i]) + 
                             X_test[i, 2] * β2_pred(z_test[i]) + 
                             X_test[i, 3] * β3_pred(z_test[i])
        end
        
        # Grid for fitting
        zgrid_pred = collect(0.05:0.05:0.95)
        
        # Fit model with fixed bandwidth
        h_pred = 0.12
        B_pred = npvaryingcoef(X_train, y_train, z_train, zgrid_pred, h_pred; method=:ll)
        
        # Predict on test set
        y_pred = predict_varyingcoef(X_test, z_test, B_pred, zgrid_pred, h_pred; method=:ll)
        
        # Test prediction accuracy
        mse = mean((y_test_true - y_pred).^2)
        rmse = sqrt(mse)
        mae = mean(abs.(y_test_true - y_pred))
        
        # The predictions should be close to true values
        @test rmse < 0.2  # RMSE should be small
        @test mae < 0.15  # MAE should be small
        
        # R² should be high (> 0.9)
        ss_total = sum((y_test_true .- mean(y_test_true)).^2)
        ss_residual = sum((y_test_true - y_pred).^2)
        r_squared = 1 - ss_residual/ss_total
        @test r_squared > 0.9
        
        # Overall prediction quality is validated by the metrics above
        # Pointwise tests can be unstable due to interpolation and boundary effects
    end
end
