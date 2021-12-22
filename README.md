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


This package implements non-parametric regression, also called local regression or kernel regression. Currently the functionality is limited to univariate regressions and to only the local constant (`localconstant`) and local linear (`locallinear`) estimators. Automatic bandwidth selection is done by leave-one-out cross validation or by optimizing the bias-corrected AICc statistic.

The two important exported convenience methods are `npregress` and `optimalbandwidth` which abstract from a lot of the implementation detail and allow you to easily switch estimators or bandwidth selection procedures.



## Examples

```julia
using NonparametricRegression

npregress

```

## Detail

- Scaled Gaussian kernel by default (`NormalKernel(h)` where `h` is the bandwidth). Kernel functionality is provided by KernelFunctions.jl.
- Computations are done in a direct and "vectorized" manner, so it is best suited for smaller data. Allocations can be substantially reduced and potentially multithreaded without changing the API much if the need is there (PRs welcome).


## Related

KernelDensity.jl is a nice package for doing kernel density estimation. 

KernelEstimators.jl is an outdated package which I found after already implementing most of this package. Consider this an updated version I guess.

LOESS.jl is a package implementing a similar but different type of local regression (loess, obviously).
