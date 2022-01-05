# https://en.wikipedia.org/wiki/Kernel_(statistics)


## these functions inspired by KernelFunctions.jl
# --- begin
abstract type AbstractKernel end


"""
    (k::AbstractKernel)(x, xg)

Evaluate the kernel for the pair of values `(x,xg)`.
"""
(k::AbstractKernel)(x, xg) = kern(k, x, xg)
kern(k::AbstractKernel, x, xg) = scaling(k) * unscaledkern(k,x,xg)
bandwidth(k::AbstractKernel) = k.h
scaling(k::AbstractKernel) = k.constant / k.h


function Base.show(io::IO, k::T) where {T<:AbstractKernel}
    return print(io, "$T (h = ", k.h, ")")
end
# --- end


# Gaussian
struct GaussianKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
GaussianKernel(h::T; constant::S=1/sqrt(2*pi)) where {T,S} = GaussianKernel{T,S}(h,constant)
unscaledkern(k::GaussianKernel, x, xg) =  exp(-((x-xg)/k.h)^2 / 2)


"""
    NormalKernel(h)

Alias of [`GaussianKernel`](@ref).
"""
const NormalKernel = GaussianKernel




# Uniform
struct UniformKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
UniformKernel(h::T; constant::S=1/2) where {T,S} = UniformKernel{T,S}(h,constant)
unscaledkern(k::UniformKernel, x::T, xg) where {T} = abs((x-xg)/k.h) <= 1 ? one(T) : zero(T)




# Epanechnikov
struct EpanechnikovKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
EpanechnikovKernel(h::T; constant::S=3/4) where {T,S} = EpanechnikovKernel{T,S}(h,constant)
unscaledkern(k::EpanechnikovKernel, x::T, xg) where {T} = abs((x-xg)/k.h) <= 1 ? 1 - ((x-xg)/k.h)^2 : zero(T)
