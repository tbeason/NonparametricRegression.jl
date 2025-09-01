# NonparametricRegression.jl

<!-- Tidyverse lifecycle badges, see https://www.tidyverse.org/lifecycle/ Uncomment or delete as needed. -->
![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![build](https://github.com/tbeason/NonparametricRegression.jl/workflows/CI/badge.svg)](https://github.com/tbeason/NonparametricRegression.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tbeason/NonparametricRegression.jl/coverage.svg?branch=main)](http://codecov.io/github/tbeason/NonparametricRegression.jl?branch=main)
<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tbeason.github.io/NonparametricRegression.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tbeason.github.io/NonparametricRegression.jl/dev)
-->


This package implements non-parametric regression, also called local regression or kernel regression. Currently the functionality is limited to univariate regressions and to only the local constant (`localconstant`) and local linear (`locallinear`,`llalphabeta`) estimators. Automatic bandwidth selection is done by leave-one-out cross validation or by optimizing the bias-corrected AICc statistic.

The two important exported convenience methods are `npregress` and `optimalbandwidth` which abstract from a lot of the implementation detail and allow you to easily switch estimators or bandwidth selection procedures.



## Examples

### Basic univariate regression
```julia
using NonparametricRegression

# Generate some data
x = rand(100)
y = sin.(2π * x) + 0.1 * randn(100)

# Estimate with automatic bandwidth selection
ŷ = npregress(x, y; method=:ll, bandwidthselection=:aicc)

# Or specify bandwidth manually
ŷ_manual = npregress(x, y, x, 0.1; method=:lc)
```

### Varying coefficient models
```julia
using NonparametricRegression

# Generate data with state-dependent coefficients
N = 500
z = rand(N)  # State variable
X = [ones(N) randn(N, 2)]  # Design matrix: intercept + 2 covariates

# True varying coefficients
β₁(z) = 1 + z
β₂(z) = 2 - z^2
β₃(z) = 0.5 * sin(2π * z)

# Generate response
y = [X[i, :]' * [β₁(z[i]), β₂(z[i]), β₃(z[i])] for i in 1:N] + 0.1 * randn(N)

# Estimate varying coefficients
zgrid = 0:0.1:1
B = npvaryingcoef(X, y, z, zgrid; method=:ll, bandwidthselection=:aicc)

# B is a 3×11 matrix where B[:, i] contains the coefficients at zgrid[i]

# Predict for new data
X_new = [ones(10) randn(10, 2)]
z_new = rand(10)
y_pred = predict_varyingcoef(X_new, z_new, B, zgrid, 0.1; method=:ll)
```

## Detail

- Scaled Gaussian kernel (`GaussianKernel`) by default (aliased by `NormalKernel(h)` where `h` is the bandwidth). Other available kernels are `UniformKernel` and `EpanechnikovKernel`. Adding a new kernel would be a relatively easy PR, see `src/kernels.jl`.
- For local linear estimation, two functions are provided. The first is `locallinear` which explicitly computes a weighted average of `y` as in `localconstant`. The second is `llalphabeta` which computes (and returns) the intercept and slope terms of the local linear regression, the intercept of which is the expected `y`. `llalphabeta` requires only one pass over the data, so is more performant than `locallinear` because computing the weights requires 2 passes, but the results are identical modulo any small numerical epsilons.
- Care was taken to make things non-allocating and performant. The package does not use the "binning" technique that other packages use (R's KernSmooth, for example), so on very large datasets there could be a performance loss relative to those packages. The package does not use multithreading, so again some performance gain could be had here if needed. PRs welcome.


## Related

KernelDensity.jl is a nice package for doing kernel density estimation. 

KernelEstimators.jl is an outdated package which I found after already implementing most of this package. Consider this an updated version I guess.

LOESS.jl is a package implementing a similar but different type of local regression (loess, obviously).
