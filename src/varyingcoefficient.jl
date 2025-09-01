########################################
########################################
##### VARYING COEFFICIENT MODELS
########################################
########################################

"""
    varyingcoefficient_lc(X, y, z, zgrid, h; kernelfun=NormalKernel)

Estimate varying coefficient model y = X*β(z) + ε using local constant (Nadaraya-Watson) approach.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `zgrid`: Grid points for evaluation (L × 1)
- `h`: Bandwidth
- `kernelfun`: Kernel function constructor (default: NormalKernel)

# Returns
- `B`: Matrix of coefficients (p × L), where B[:,i] = β(zgrid[i])
"""
function varyingcoefficient_lc(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, zgrid::AbstractVector, h; kernelfun=NormalKernel)
    K = kernelfun(h)
    N, p = size(X)
    @assert N == length(y) == length(z) "Dimensions of X, y, and z must be consistent"
    L = length(zgrid)
    
    B = zeros(p, L)  # coefficient matrix
    
    @inbounds for i in 1:L
        # Compute kernel weights
        w = zeros(N)
        for j in 1:N
            w[j] = unscaledkern(K, z[j], zgrid[i])
        end
        
        # Weighted least squares: β = (X'WX)^(-1)X'Wy
        XtW = X' * Diagonal(w)
        XtWX = XtW * X
        XtWy = XtW * y
        
        # Solve for coefficients
        B[:, i] = XtWX \ XtWy
    end
    
    return B
end


"""
    varyingcoefficient_ll(X, y, z, zgrid, h; kernelfun=NormalKernel)

Estimate varying coefficient model y = X*β(z) + ε using local linear approach.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `zgrid`: Grid points for evaluation (L × 1)
- `h`: Bandwidth
- `kernelfun`: Kernel function constructor (default: NormalKernel)

# Returns
- `B`: Matrix of coefficients (p × L), where B[:,i] = β(zgrid[i])
"""
function varyingcoefficient_ll(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, zgrid::AbstractVector, h; kernelfun=NormalKernel)
    K = kernelfun(h)
    N, p = size(X)
    @assert N == length(y) == length(z) "Dimensions of X, y, and z must be consistent"
    L = length(zgrid)
    
    B = zeros(p, L)  # coefficient matrix
    
    @inbounds for i in 1:L
        # Compute kernel weights and z differences
        w = zeros(N)
        zdiff = zeros(N)
        for j in 1:N
            w[j] = unscaledkern(K, z[j], zgrid[i])
            zdiff[j] = z[j] - zgrid[i]
        end
        
        # Augmented design matrix: [X, X.*(z-zgrid[i])]
        Xaug = hcat(X, X .* zdiff)
        
        # Weighted least squares
        XtW = Xaug' * Diagonal(w)
        XtWX = XtW * Xaug
        XtWy = XtW * y
        
        # Solve for coefficients and extract level coefficients
        coeffs = XtWX \ XtWy
        B[:, i] = coeffs[1:p]  # Level coefficients β(zgrid[i])
    end
    
    return B
end


"""
    varyingcoefficient_aicc(X, y, z, h; kernelfun=NormalKernel, method=:lc)

Compute AICc for varying coefficient model with given bandwidth.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `h`: Bandwidth
- `kernelfun`: Kernel function constructor (default: NormalKernel)
- `method`: Estimation method (:lc for local constant, :ll for local linear)

# Returns
- `aicc`: AICc value
"""
function varyingcoefficient_aicc(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, h; kernelfun=NormalKernel, method=:lc)
    K = kernelfun(h)
    N, p = size(X)
    @assert N == length(y) == length(z) "Dimensions of X, y, and z must be consistent"
    
    sigmasq = 0.0
    traceH = 0.0
    
    @inbounds for i in 1:N
        if method == :lc
            # Local constant
            w = zeros(N)
            for j in 1:N
                w[j] = unscaledkern(K, z[j], z[i])
            end
            
            XtW = X' * Diagonal(w)
            XtWX = XtW * X
            XtWy = XtW * y
            
            beta_i = XtWX \ XtWy
            yhat_i = dot(X[i, :], beta_i)
            
            # Contribution to trace(H)
            Hi = X[i, :]' * (XtWX \ (XtW[:, i]))
            traceH += Hi
            
        elseif method == :ll
            # Local linear
            w = zeros(N)
            zdiff = zeros(N)
            for j in 1:N
                w[j] = unscaledkern(K, z[j], z[i])
                zdiff[j] = z[j] - z[i]
            end
            
            Xaug = hcat(X, X .* zdiff)
            XtW = Xaug' * Diagonal(w)
            XtWX = XtW * Xaug
            XtWy = XtW * y
            
            coeffs = XtWX \ XtWy
            yhat_i = dot(X[i, :], coeffs[1:p])
            
            # Contribution to trace(H)
            Xi_aug = vcat(X[i, :], zeros(p))  # Only level terms for prediction
            Hi = dot(Xi_aug, XtWX \ (XtW[:, i]))
            traceH += Hi
        else
            error("Unknown method: $method")
        end
        
        sigmasq += (y[i] - yhat_i)^2 / N
    end
    
    aicc = AICc(sigmasq, traceH, N)
    return aicc
end


"""
    varyingcoefficientweights(X, z, zgrid, h; kernelfun=NormalKernel, method=:lc)

Compute weight matrix for varying coefficient model.

# Arguments
- `X`: Design matrix (N × p)
- `z`: State variable vector (N × 1)
- `zgrid`: Grid points for evaluation (L × 1)
- `h`: Bandwidth
- `kernelfun`: Kernel function constructor (default: NormalKernel)
- `method`: Estimation method (:lc for local constant, :ll for local linear)

# Returns
- `W`: Weight matrix (N × L) where W[:,i] contains weights for prediction at zgrid[i]
"""
function varyingcoefficientweights(X::AbstractMatrix, z::AbstractVector, zgrid::AbstractVector, h; kernelfun=NormalKernel, method=:lc)
    K = kernelfun(h)
    N, p = size(X)
    @assert N == length(z) "Dimensions of X and z must be consistent"
    L = length(zgrid)
    
    W = zeros(N, L)
    
    @inbounds for i in 1:L
        if method == :lc
            # Local constant weights
            w = zeros(N)
            for j in 1:N
                w[j] = unscaledkern(K, z[j], zgrid[i])
            end
            
            XtW = X' * Diagonal(w)
            XtWX = XtW * X
            
            # Weight matrix: W = X(X'WX)^(-1)X'W
            for j in 1:N
                W[j, i] = dot(X[j, :], XtWX \ (XtW[:, j]))
            end
            
        elseif method == :ll
            # Local linear weights
            w = zeros(N)
            zdiff = zeros(N)
            for j in 1:N
                w[j] = unscaledkern(K, z[j], zgrid[i])
                zdiff[j] = z[j] - zgrid[i]
            end
            
            Xaug = hcat(X, X .* zdiff)
            XtW = Xaug' * Diagonal(w)
            XtWX = XtW * Xaug
            
            # Extract level weights only
            for j in 1:N
                Xj_aug = vcat(X[j, :], X[j, :] .* zdiff[j])
                wj = XtWX \ (XtW[:, j])
                W[j, i] = dot(X[j, :], wj[1:p])
            end
        else
            error("Unknown method: $method")
        end
    end
    
    return W
end


########################################
########################################
##### BANDWIDTH SELECTION
########################################
########################################

"""
    leaveoneoutCV_varyingcoef(X, y, z, h; kernelfun=NormalKernel, method=:lc)

Leave-one-out cross-validation for varying coefficient models.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `h`: Bandwidth
- `kernelfun`: Kernel function constructor (default: NormalKernel)
- `method`: Estimation method (:lc for local constant, :ll for local linear)

# Returns
- `mse`: Mean squared error from LOOCV
"""
function leaveoneoutCV_varyingcoef(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, h; kernelfun=NormalKernel, method=:lc)
    N = size(X, 1)
    @assert N == length(y) == length(z) "Dimensions of X, y, and z must be consistent"
    
    mse = 0.0
    inds = 1:N
    
    @inbounds for i in 1:N
        # Leave out observation i
        train_inds = filter(!=(i), inds)
        X_train = view(X, train_inds, :)
        y_train = view(y, train_inds)
        z_train = view(z, train_inds)
        
        # Predict at z[i]
        if method == :lc
            B = varyingcoefficient_lc(X_train, y_train, z_train, [z[i]], h; kernelfun)
        elseif method == :ll
            B = varyingcoefficient_ll(X_train, y_train, z_train, [z[i]], h; kernelfun)
        else
            error("Unknown method: $method")
        end
        
        yhat_i = dot(X[i, :], B[:, 1])
        mse += (y[i] - yhat_i)^2 / N
    end
    
    return mse
end


########################################
########################################
##### HIGH-LEVEL API
########################################
########################################

"""
    npvaryingcoef(X, y, z, zgrid, h; method=:lc, kernelfun=NormalKernel)

Estimate varying coefficient model with given bandwidth.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `zgrid`: Grid points for evaluation (L × 1)
- `h`: Bandwidth
- `method`: Estimation method (:lc for local constant, :ll for local linear)
- `kernelfun`: Kernel function constructor (default: NormalKernel)

# Returns
- `B`: Matrix of coefficients (p × L), where B[:,i] = β(zgrid[i])
"""
function npvaryingcoef(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, zgrid::AbstractVector, h; method=:lc, kernelfun=NormalKernel)
    if method == :lc || method == :localconstant
        return varyingcoefficient_lc(X, y, z, zgrid, h; kernelfun)
    elseif method == :ll || method == :locallinear
        return varyingcoefficient_ll(X, y, z, zgrid, h; kernelfun)
    else
        error("Unknown method: $method. Use :lc (local constant) or :ll (local linear)")
    end
end


"""
    npvaryingcoef(X, y, z, zgrid=z; method=:lc, bandwidthselection=:aicc, kernelfun=NormalKernel, hLB=silvermanbw(z)/100, hUB=silvermanbw(z)*100)

Estimate varying coefficient model with automatic bandwidth selection.

# Arguments
- `X`: Design matrix (N × p)
- `y`: Response vector (N × 1)
- `z`: State variable vector (N × 1)
- `zgrid`: Grid points for evaluation (default: z)
- `method`: Estimation method (:lc for local constant, :ll for local linear)
- `bandwidthselection`: Method for bandwidth selection (:aicc or :loocv)
- `kernelfun`: Kernel function constructor (default: NormalKernel)
- `hLB`, `hUB`: Lower and upper bounds for bandwidth search

# Returns
- `B`: Matrix of coefficients (p × L), where B[:,i] = β(zgrid[i])
"""
function npvaryingcoef(X::AbstractMatrix, y::AbstractVector, z::AbstractVector, zgrid::AbstractVector=z; 
                      method=:lc, bandwidthselection=:aicc, kernelfun=NormalKernel, 
                      hLB=silvermanbw(z)/100, hUB=silvermanbw(z)*100)
    
    # Select optimal bandwidth
    if bandwidthselection == :aicc || bandwidthselection == :aic
        objfun_aicc(h) = varyingcoefficient_aicc(X, y, z, h; kernelfun, method)
        h_opt = optimize(objfun_aicc, hLB, hUB)
    elseif bandwidthselection == :loocv || bandwidthselection == :cv
        objfun_cv(h) = leaveoneoutCV_varyingcoef(X, y, z, h; kernelfun, method)
        h_opt = optimize(objfun_cv, hLB, hUB)
    else
        error("Unknown bandwidth selection method: $bandwidthselection")
    end
    
    # Estimate with optimal bandwidth
    return npvaryingcoef(X, y, z, zgrid, h_opt; method, kernelfun)
end


"""
    predict_varyingcoef(X_new, z_new, B, zgrid, h; method=:lc, kernelfun=NormalKernel)

Predict responses for new observations using fitted varying coefficient model.

# Arguments
- `X_new`: Design matrix for new observations (M × p)
- `z_new`: State variable for new observations (M × 1)
- `B`: Fitted coefficient matrix from npvaryingcoef (p × L)
- `zgrid`: Grid points used in estimation (L × 1)
- `h`: Bandwidth used in estimation
- `method`: Estimation method used (:lc or :ll)
- `kernelfun`: Kernel function constructor

# Returns
- `y_pred`: Predicted values (M × 1)
"""
function predict_varyingcoef(X_new::AbstractMatrix, z_new::AbstractVector, B::AbstractMatrix, 
                           zgrid::AbstractVector, h; method=:lc, kernelfun=NormalKernel)
    M, p = size(X_new)
    @assert M == length(z_new) "Dimensions of X_new and z_new must be consistent"
    @assert size(B, 1) == p "Number of coefficients must match number of covariates"
    @assert size(B, 2) == length(zgrid) "Coefficient matrix dimensions must match zgrid"
    
    y_pred = zeros(M)
    
    # For each new observation
    @inbounds for i in 1:M
        # Interpolate coefficients at z_new[i]
        beta_i = zeros(p)
        for j in 1:p
            if method == :lc || method == :localconstant
                beta_i[j] = localconstant(zgrid, B[j, :], [z_new[i]], h; kernelfun)[1]
            elseif method == :ll || method == :locallinear
                beta_i[j] = locallinear(zgrid, B[j, :], [z_new[i]], h; kernelfun)[1]
            else
                error("Unknown method: $method")
            end
        end
        
        y_pred[i] = dot(X_new[i, :], beta_i)
    end
    
    return y_pred
end