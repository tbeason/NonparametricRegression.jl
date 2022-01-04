# https://en.wikipedia.org/wiki/Kernel_(statistics)


## these functions inspired by KernelFunctions.jl
# --- begin
abstract type AbstractKernel end

## Allows to iterate over kernels
Base.length(::AbstractKernel) = 1
Base.iterate(k::AbstractKernel) = (k, nothing)
Base.iterate(k::AbstractKernel, ::Any) = nothing


(k::AbstractKernel)(x, xg) = kern(k, x, xg)
scaling(k::AbstractKernel) = k.constant / k.h
bandwidth(k::AbstractKernel) = k.h

function Base.show(io::IO, k::T) where {T<:AbstractKernel}
    return print(io, "$T (h = ", k.h, ", scaling = ",scaling(k),")")
end
# --- end


# Gaussian
struct GaussianKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
GaussianKernel(h::T, s::S=1/sqrt(2*pi)) where {T,S} = GaussianKernel{T,S}(h,s)
kern(k::GaussianKernel, x, xg) =  exp(-((x-xg)/k.h)^2 / 2)





# Uniform
struct UniformKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
UniformKernel(h::T, s::S=1/2) where {T,S} = UniformKernel{T,S}(h,s)
kern(k::UniformKernel, x::T, xg) where {T} = abs((x-xg)/k.h) <= 1 ? one(T) : zero(T)




# Epanechnikov
struct EpanechnikovKernel{T<:Real,S<:Real} <: AbstractKernel
    h::T
    constant::S
end
EpanechnikovKernel(h::T, s::S=3/4) where {T,S} = EpanechnikovKernel{T,S}(h,s)
kern(k::EpanechnikovKernel, x::T, xg) where {T} = abs((x-xg)/k.h) <= 1 ? 1 - ((x-xg)/k.h)^2 : zero(T)
