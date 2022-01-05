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

```julia
using NonparametricRegression

npregress

```

## Detail

- Scaled Gaussian kernel (`GaussianKernel`) by default (aliased by `NormalKernel(h)` where `h` is the bandwidth). Other available kernels are `UniformKernel` and `EpanechnikovKernel`. Adding a new kernel would be a relatively easy PR, see `src/kernels.jl`.
- For local linear estimation, two functions are provided. The first is `locallinear` which explicitly computes a weighted average of `y` as in `localconstant`. The second is `llalphabeta` which computes (and returns) the intercept and slope terms of the local linear regression, the intercept of which is the expected `y`. `llalphabeta` requires only one pass over the data, so is more performant than `locallinear` because computing the weights requires 2 passes, but the results are identical modulo any small numerical epsilons.
- Care was taken to make things non-allocating and performant. The package does not use the "binning" technique that other packages use (R's KernSmooth, for example), so on very large datasets there could be a performance loss relative to those packages. The package does not use multithreading, so again some performance gain could be had here if needed. PRs welcome.


## Related

KernelDensity.jl is a nice package for doing kernel density estimation. 

KernelEstimators.jl is an outdated package which I found after already implementing most of this package. Consider this an updated version I guess.

LOESS.jl is a package implementing a similar but different type of local regression (loess, obviously).
