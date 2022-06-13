"""
NonparametricRegression.jl implements local regression methods, also known as kernel regression.

Type `? npregress` for more detail.
"""
module NonparametricRegression

using LinearAlgebra
using Statistics
using StaticArrays
using DocStringExtensions


export NormalKernel, GaussianKernel, UniformKernel, EpanechnikovKernel, bandwidth
export localconstant, localconstantweights
export locallinear, llalphabeta, locallinearweights
export optimalbandwidth, leaveoneoutCV, optimizeAICc
export npregress

include("kernels.jl")
include("univariateopt.jl")


########################################
########################################
##### UTILITY FUNCTIONS
########################################
########################################


AICc(sigmasq,trH,Ny) = log(sigmasq) + 1 + 2*(trH + 1)/(Ny-trH-2)


## KDE
# this function is only used internally for trimmed LOOCV to prevent adding a dep on KernelDensity.jl
silvermanbw(x,alpha=0.9) = alpha * min(std(x), only(diff(quantile(x, [0.25, 0.75])))/1.34) * length(x)^(-1/5)

function densityestimator(x,xgrid=x; h=silvermanbw(x), kernelfun=NormalKernel)
	K = kernelfun(h)
	N = length(x)
	L = length(xgrid)
	f = zeros(L) 	# preallocate
	@inbounds for i in 1:L
		for j in 1:N
			f[i] += unscaledkern(K,x[j],xgrid[i]) 
		end 
		f[i] *= scaling(K) / N
	end
	return f
end











########################################
########################################
##### SPECIFIC NONPARAMETRIC REGRESSION METHODS
########################################
########################################

## LOCAL CONSTANT aka NADARAYA-WATSON ESTIMATOR
function localconstant(x,y,xgrid, h; kernelfun=NormalKernel)
	K = kernelfun(h)
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."
	L = length(xgrid)
	m = zeros(L)	# m is expected y, at points xgrid
	@inbounds for i in 1:L
		wu = 0.0
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			wu += ku * y[j]
			wsum += ku
		end
		m[i] = wu / wsum
	end
	return m
end

function localconstant_aicc(x,y,h; kernelfun=NormalKernel)
	K = kernelfun(h)
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."

	sigmasq = 0.0
	traceH = 0.0
	@inbounds for i in 1:Nx
		wu = 0.0
		wsum = 0.0
		tH = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],x[i])
			wu += ku * y[j]
			wsum += ku
			if j == i
				tH = ku
			end
		end
		yp = wu / wsum
		sigmasq += (y[i] - yp)^2 / Nx
		traceH += tH / wsum
	end
	aicc = AICc(sigmasq,traceH,Nx)
	return aicc
end

function localconstantweights(x,xgrid, h; kernelfun=NormalKernel, normalizeweights::Bool=true)
	K = kernelfun(h)
	Nx = length(x)
	L = length(xgrid)
	W = zeros(Nx,L)	# m is expected y, at points xgrid
	@inbounds for i in 1:L
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			W[j,i] = ku
			wsum += ku
		end
		if normalizeweights
			W[:,i] ./= wsum
		end
	end
	return W
end






## LOCAL LINEAR REGRESSION
function locallinear(x,y,xgrid, h; kernelfun=NormalKernel)
	K = kernelfun(h)
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."
	L = length(xgrid)
	m = zeros(L)	# m is expected y, at points xgrid
	@inbounds for i in 1:L
		s1sum = 0.0
		s2sum = 0.0
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			xmx = x[j]- xgrid[i]
			s1sum += ku*xmx
			s2sum += ku*xmx^2
			wsum += ku
		end
		s1sum /= wsum
		s2sum /= wsum

		wu = 0.0
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			xmx = x[j]- xgrid[i]
			w = ku * (s2sum - s1sum*xmx)
			wu += w * y[j]
			wsum += w
		end
		m[i] = wu / wsum
	end
	return m
end



function locallinear_aicc(x,y, h; kernelfun=NormalKernel)
	K = kernelfun(h)
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."
	
	sigmasq = 0.0
	traceH = 0.0
	@inbounds for i in 1:Nx
		s1sum = 0.0
		s2sum = 0.0
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],x[i])
			xmx = x[j]- x[i]
			s1sum += ku*xmx
			s2sum += ku*xmx^2
			wsum += ku
		end
		s1sum /= wsum
		s2sum /= wsum

		wu = 0.0
		wsum = 0.0
		tH = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],x[i])
			xmx = x[j]- x[i]
			w = ku * (s2sum - s1sum*xmx)
			wu += w * y[j]
			wsum += w
			if j == i
				tH = w
			end
		end
		yp = wu / wsum
		sigmasq += (y[i] - yp)^2 / Nx
		traceH += tH / wsum
	end
	aicc = AICc(sigmasq,traceH,Nx)
	return aicc
end



function llalphabeta(x,y,xgrid, h; kernelfun=NormalKernel)
	K = kernelfun(h)
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."
	L = length(xgrid)
	avec = zeros(L)
	bvec = zeros(L)
	@inbounds for i in 1:L
		l = SMatrix{2,2}(0.0,0.0,0.0,0.0)
		r = SVector(0.0,0.0)
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			zj = SVector(1.0,x[j]- xgrid[i])
			l += ku*zj*zj'
			r += ku*zj*y[j]
			wsum += ku
		end
		a,b = l/wsum \ r/wsum
		avec[i] = a
		bvec[i] = b
	end
	return avec,bvec
end


function locallinearweights(x,xgrid, h; kernelfun=NormalKernel, normalizeweights::Bool=true)
	K = kernelfun(h)
	Nx = length(x)
	L = length(xgrid)
	W = zeros(Nx,L)
	for i in 1:L
		s1sum = 0.0
		s2sum = 0.0
		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			xmx = x[j]- xgrid[i]
			s1sum += ku*xmx
			s2sum += ku*xmx^2
			wsum += ku
		end
		s1sum /= wsum
		s2sum /= wsum

		wsum = 0.0
		for j in 1:Nx
			ku = unscaledkern(K,x[j],xgrid[i])
			xmx = x[j]- xgrid[i]
			w = ku * (s2sum - s1sum*xmx)
			W[j,i] = w
			wsum += w
		end
		if normalizeweights
			W[:,i] ./= wsum
		end
	end
	return W
end

########################################
########################################
##### BANDWIDTH SELECTION 
########################################
########################################
## LOOCV
function leaveoneoutCV_mse(h,x,y; kernelfun=NormalKernel, method=:lc, trimmed=true)
	if trimmed
		kde = densityestimator(x)
	end
	
	Nx = length(x)
	@assert Nx == length(y) "Length of `x` and `y` must agree."
	mse = 0.0
	inds = eachindex(x)
	@inbounds for i in 1:Nx
		xtmp = view(x,filter(!=(i),inds))
		ytmp = view(y,filter(!=(i),inds))
		# pred = estimatorfun(xtmp,ytmp,x[i], h, kernelfun)
		if method in (:nw,:lc,:localconstant)
			pred = only(localconstant(xtmp,ytmp,x[i], h; kernelfun))
		elseif method in (:ll,:locallinear,:llalphabeta)
			pred = only(first(llalphabeta(xtmp,ytmp,x[i], h; kernelfun)))
		else
			error("Unknown method supplied")
		end
		if trimmed
			mse += (y[i] - pred)^2 * kde[i] / Nx
		else
			mse += (y[i] - pred)^2 / Nx
		end
	end

	return mse
end




function leaveoneoutCV(x, y; kernelfun=NormalKernel, method=:lc, hLB = silvermanbw(y)/100, hUB = silvermanbw(y)*100,trimmed=true)
	objfun(h) = leaveoneoutCV_mse(h,x,y; kernelfun, method, trimmed)
	opt = optimize(objfun,hLB,hUB)
	# @assert opt.converged "Convergence failed, cannot find optimal bandwidth."
	return opt
end


function estimatorAICc(h, x, y; kernelfun=NormalKernel, method=:lc)
    if method in (:nw,:lc,:localconstant)
		return localconstant_aicc(x, y, h; kernelfun)
	elseif method in (:ll,:locallinear,:llalphabeta)
		return locallinear_aicc(x, y, h; kernelfun)
	else
		error("Unknown method supplied")
	end
end

function optimizeAICc(x, y; kernelfun=NormalKernel, method=:lc, hLB = silvermanbw(y)/100, hUB = silvermanbw(y)*100)
    objfun(h) = estimatorAICc(h,x,y; kernelfun, method)
	opt = optimize(objfun,hLB,hUB)
	# @assert opt.converged "Convergence failed, cannot find optimal bandwidth."
	return opt
end








########################################
########################################
##### CONVENIENCE METHODS 
########################################
########################################
"""
$SIGNATURES

Search for the optimal bandwidth to use for the local regression of `y` against `x` using `method ∈ (:lc,:ll)`.

The keyword argument `bandwidthselection` should be `:aicc` for the bias-corrected AICc method or `:loocv` for leave-one-out cross validation. `hLB` and `hUB` are the lower and upper bounds used when searching for the optimal bandwidth.
"""
function optimalbandwidth(x, y; kernelfun=NormalKernel, method=:lc, bandwidthselection=:aicc, hLB = silvermanbw(y)/100, hUB = silvermanbw(y)*100)
    # automatically select bandwidth
    if bandwidthselection in (:cv, :loocv, :leaveoneoutcv)
        h = leaveoneoutCV(x,y; kernelfun, method, hLB, hUB)
    elseif bandwidthselection in (:aicc, :aic)
        h = optimizeAICc(x,y; kernelfun, method, hLB, hUB)
    else
        error("Unknown bandwidth selection procedure supplied. See `? optimalbandwidth` for valid inputs.")
    end
    return h
end


"""
$SIGNATURES

Estimate a local regression of `y` against `x` evaluated at the values `xgrid` using bandwidth `h`.

The keyword argument `method` can be `:lc` for a local constant estimator (Nadaraya-Watson) or `:ll` for a local linear estimator.

The keyword argument `kernelfun` should be a function that constructs a kernel with a given bandwidth. Defaults to `NormalKernel` (defined and exported by this package.)
"""
function npregress(x,y,xgrid,h; kernelfun=NormalKernel, method=:lc)
    if method in (:nw,:lc,:localconstant)
		return localconstant(x,y,xgrid,h; kernelfun)
	elseif method in (:ll,:locallinear)
		return locallinear(x,y,xgrid,h; kernelfun)
	else
		error("Unknown regression method supplied. See `? npregress` for valid inputs.")
	end
end


"""
$SIGNATURES

Estimate a local regression of `y` against `x`, possibly evaluated at the values `xgrid`, and automatically select the optimal bandwidth.

The keyword argument `method` can be `:lc` for a local constant estimator (Nadaraya-Watson) or `:ll` for a local linear estimator.

The keyword argument `bandwidthselection` should be `:aicc` for the bias-correct AICc method or `:loocv` for leave-one-out cross validation. `hLB` and `hUB` are the lower and upper bounds used when searching for the optimal bandwidth.

The keyword argument `kernelfun` should be a function that constructs a kernel with a given bandwidth. Defaults to `NormalKernel` (defined and exported by this package.)
"""
function npregress(x, y, xgrid=x; kernelfun=NormalKernel, method=:lc, bandwidthselection=:aicc, hLB = silvermanbw(y)/100, hUB = silvermanbw(y)*100)
    h = optimalbandwidth(x,y; kernelfun, method, bandwidthselection, hLB, hUB)
    return npregress(x,y,xgrid,h; kernelfun, method)
end





########################################
########################################
##### CURRENTLY UNUSED METHODS (BUT MAYBE RELEVANT FOR FUTURE WORK)
########################################
########################################
## commented out to increase coverage :)

# function transformedresiduals(y::AbstractVector{T1},H::AbstractMatrix{T2}; method=:HC3) where {T1,T2}
# 	r = y .- H*y
# 	if method == :HC0
# 		return r
# 	elseif method == :HC1
# 		error("HC1 not implemented")
# 	elseif method == :HC2
# 		Hd = diag(H)
# 		return r./sqrt.(1 .- Hd)
# 	elseif method == :HC3
# 		Hd = diag(H)
# 		return r./(1 .- Hd)
# 	else
# 		error("unknown method")
# 	end
# end



end # module
