"""
NonparametricRegression.jl implements local regression methods, also known as kernel regression.

Type `? npregress` for more detail.
"""
module NonparametricRegression

using LinearAlgebra
using Statistics
using KernelFunctions
using Optim

export NormalKernel
export localconstant, localconstantweights
export locallinear, locallinearweights, llalphabeta
export optimalbandwidth, leaveoneoutCV, optimizeAICc
export npregress


########################################
########################################
##### UTILITY FUNCTIONS
########################################
########################################

## MANIPULATING MATRICES
normalizecols(X::AbstractMatrix{T}) where {T} = normalizecols!(copy(X))
normalizecols!(X::AbstractMatrix{T}) where {T} = X ./= sum(X;dims=1)
function zerodiagonal!(X::AbstractMatrix{T}) where {T}
    @inbounds for i in 1:size(X,1)
        X[i,i] = zero(T)
    end
end

## KERNEL UTILITIES
# extractbw(k) = 1 / only(k.kernel.transform.s)

# note to users: treat h as the standard deviation of the data
NormalKernel(h) = ScaledKernel(SqExponentialKernel() ∘ ScaleTransform(1/h),1/(sqrt(2*π)*h))

## KDE
# this function is only used internally for trimmed LOOCV to prevent adding a dep on KernelDensity.jl
function densityestimator(x,xgrid=x; h=silvermanbw(x), kernelfun::Function=NormalKernel)
	K = kernelfun(h)
	W = kernelmatrix(K,x,xgrid)
	f = mean(W;dims=1)
	return vec(f)
end


silvermanbw(x,alpha=0.9) = alpha * min(std(x), only(diff(quantile(x, [0.25, 0.75])))/1.34) * length(x)^(-1/5)










########################################
########################################
##### SPECIFIC NONPARAMETRIC REGRESSION METHODS
########################################
########################################

## LOCAL CONSTANT aka NADARAYA-WATSON ESTIMATOR
function localconstantweights(x,xgrid, h; kernelfun::Function=NormalKernel, normalizeweights::Bool=true)
	K = kernelfun(h)
	W = kernelmatrix(K,x,xgrid)
    !normalizeweights && return W
	normalizecols!(W)
	return W
end

function localconstant(x,y,xgrid, h; kernelfun::Function=NormalKernel, normalizeweights::Bool=true)
	W = localconstantweights(x,xgrid, h; kernelfun, normalizeweights)
	return W'*y
end







## LOCAL LINEAR REGRESSION
function locallinearweights(x,xgrid, h; kernelfun::Function=NormalKernel, normalizeweights::Bool=true)
	K = kernelfun(h)
	Wk = kernelmatrix(K,x,xgrid)
	xmx = x .- xgrid'
	S1 = vec(sum(Wk .* xmx;dims=1))
	S2 = vec(sum(Wk .* xmx.^2;dims=1))
	W = Wk .* (S2' .- S1' .* xmx)
	!normalizeweights && return W
	normalizecols!(W)
	return W
end

function locallinear(x,y,xgrid, h; kernelfun::Function=NormalKernel, normalizeweights::Bool=true)
	W = locallinearweights(x,xgrid, h; kernelfun, normalizeweights)
	return W'*y
end


function llalphabeta(x,y,xgrid, h; kernelfun::Function=NormalKernel)
	K = kernelfun(h)
	W = kernelmatrix(K,x,xgrid)
	L = length(xgrid)
	N = length(x)
	avec = zeros(L)
	bvec = zeros(L)
	vec1 = ones(N)
	for i in 1:L
		l = zeros(2,2)
		r = zeros(2)
		for j in 1:N
			zj = [1,x[j]- xgrid[i]]
			l += W[j,i]*zj*zj'
			r += W[j,i]*zj*y[j]
		end
		a,b = l \ r
		avec[i] = a
		bvec[i] = b
	end
	return avec,bvec
end


########################################
########################################
##### BANDWIDTH SELECTION 
########################################
########################################
## LOOCV
function leaveoneoutCV_mse(h,x,y; kernelfun::Function=NormalKernel, method=:lc)
	kde = densityestimator(x)
	if method in (:nw,:lc,:localconstant)
		W=localconstantweights(x,x, h; kernelfun, normalizeweights=false)
		zerodiagonal!(W)
		normalizecols!(W)
		mse = mean((y .- W'*y).^2 .* kde)
	elseif method in (:ll,:locallinear)
        N = length(x)
        mse = 0.0
        inds = eachindex(x)
        for i in 1:N
            xtmp = view(x,filter(!=(i),inds))
            ytmp = view(y,filter(!=(i),inds))
		    pred = only(locallinear(xtmp,ytmp,[x[i]], h; kernelfun))
            mse += (y[i] - pred)^2 * kde[i] / N
        end
	else
		error("Unknown method supplied")
	end
	return mse
end

function leaveoneoutCV(x, y; kernelfun::Function=NormalKernel, method=:lc, hLB = 0.01, hUB = 10.0)
	objfun(h) = leaveoneoutCV_mse(h,x,y; kernelfun, method)
	opt = optimize(objfun,hLB,hUB)
    # should check convergence before returning...
	return Optim.minimizer(opt)
end

## AICc
function AICc(y::AbstractVector{T1},H::AbstractMatrix{T2}) where {T1,T2}
	# H == W' for nw and ll, projection matrix for linear regressions
	yhat = H*y
	sigmasq = mean(abs2,y - yhat)
	trH = tr(H)
	aicc = log(sigmasq) + 1 + 2*(trH + 1)/(length(y)-trH-2)
	return aicc
end

function estimatorAICc(h, x, y; kernelfun::Function=NormalKernel, method=:lc)
    if method in (:nw,:lc,:localconstant)
		W=localconstantweights(x, x, h; kernelfun)
	elseif method in (:ll,:locallinear)
		W = locallinearweights(x, x, h; kernelfun)
	else
		error("Unknown method supplied")
	end
    return AICc(y,W')
end

function optimizeAICc(x, y; kernelfun::Function=NormalKernel, method=:lc, hLB = 0.01, hUB = 10.0)
    objfun(h) = estimatorAICc(h,x,y; kernelfun, method)
	opt = optimize(objfun,hLB,hUB)
    # should check convergence before returning...
	return Optim.minimizer(opt)
end








########################################
########################################
##### CONVENIENCE METHODS 
########################################
########################################
function optimalbandwidth(x, y; kernelfun::Function=NormalKernel, method=:lc, bandwidthselection=:aicc, hLB=1e-2, hUB=10.0)
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



function npregress(x,y,xgrid,h; kernelfun::Function=NormalKernel, method=:lc)
    if method in (:nw,:lc,:localconstant)
		return localconstant(x,y,xgrid,h; kernelfun)
	elseif method in (:ll,:locallinear)
		return locallinear(x,y,xgrid,h; kernelfun)
	else
		error("Unknown regression method supplied. See `? npregress` for valid inputs.")
	end
end


function npregress(x, y, xgrid=x; kernelfun::Function=NormalKernel, method=:lc, bandwidthselection=:aicc, hLB=1e-2, hUB=10.0)
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
