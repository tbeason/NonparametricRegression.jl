
# borrowed from KernelDensity.jl
# https://github.com/JuliaStats/KernelDensity.jl/blob/master/src/univariate.jl

"""
optimize(f, x_lower, x_upper; iterations=1000, rel_tol=nothing, abs_tol=nothing)

Minimize the function `f` in the interval `x_lower..x_upper`, using the
[golden-section search](https://en.wikipedia.org/wiki/Golden-section_search).
Return an approximate minimum `x̃` or error if such approximate minimum cannot be found.

This algorithm assumes that `-f` is unimodal on the interval `x_lower..x_upper`,
that is to say, there exists a unique `x` in `x_lower..x_upper` such that `f` is
decreasing on `x_lower..x` and increasing on `x..x_upper`.

`rel_tol` and `abs_tol` determine the relative and absolute tolerance, that is
to say, the returned value `x̃` should differ from the actual minimum `x` at most
`abs_tol + rel_tol * abs(x̃)`.
If not manually specified, `rel_tol` and `abs_tol` default to `sqrt(eps(T))` and
`eps(T)` respectively, where `T` is the floating point type of `x_lower` and `x_upper`.

`iterations` determines the maximum number of iterations allowed before convergence.

This is a private, unexported function, used internally to select the optimal bandwidth
automatically.
"""
function optimize(f, x_lower, x_upper; iterations=1000, rel_tol=nothing, abs_tol=nothing)

if x_lower > x_upper
    error("x_lower must be less than x_upper")
end

T = promote_type(typeof(x_lower/1), typeof(x_upper/1))
rtol = something(rel_tol, sqrt(eps(T)))
atol = something(abs_tol, eps(T))

function midpoint_and_convergence(lower, upper)
    midpoint = (lower + upper) / 2
    tol = atol + rtol * midpoint
    midpoint, (upper - lower) <= 2tol
end

invphi::T = 0.5 * (sqrt(5) - 1)
invphisq::T = 0.5 * (3 - sqrt(5))

a::T, b::T = x_lower, x_upper
h = b - a
c = a + invphisq * h
d = a + invphi * h

fc, fd = f(c), f(d)

for _ in 1:iterations
    h *= invphi
    if fc < fd
        m, converged = midpoint_and_convergence(a, d)
        converged && return m
        b = d
        d, fd = c, fc
        c = a + invphisq * h
        fc = f(c)
    else
        m, converged = midpoint_and_convergence(c, b)
        converged && return m
        a = c
        c, fc = d, fd
        d = a + invphi * h
        fd = f(d)
    end
end

error("Reached maximum number of iterations without convergence.")
end