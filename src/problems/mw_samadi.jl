using LinearAlgebra
using Printf
using Statistics

function mw(A::Matrix{Float64}, B::Matrix{Float64}, d::Int, eta::Float64, T::Int)
    """
    MW method for fair dimensionality reduction between two groups
    
    Parameters:
    A, B: Data matrices (rows are data points, columns are features)
    d: Target dimension
    eta: MW learning rate
    T: Number of iterations
    
    Returns:
    P: Average projection matrix
    z: Maximum loss between groups
    P_last: Last iteration projection matrix
    z_last: Last iteration loss
    """
    
    println("MW method is called")
    
    # Compute covariance matrices
    covA = A' * A
    covB = B' * B
    
    # Get sizes
    m_A, n = size(A)
    m_B = size(B, 1)
    
    # Compute optimal d-dimensional approximations
    # Note: In Julia, we need to implement optApprox or use SVD
    Ahat = optApprox(A, d)
    alpha = norm(Ahat)^2  # Frobenius norm squared
    
    Bhat = optApprox(B, d)
    beta = norm(Bhat)^2
    
    # MW initialization
    w_1 = 0.5
    w_2 = 0.5
    
    # Initialize P as sum of all P_t
    P = zeros(n, n)
    
    # Record progress
    record = Vector{Vector{Any}}()
    push!(record, ["iteration", "w_1", "w_2", "loss A", "loss B", "loss A by average", "loss B by average"])

    P_temp=zeros(n, n)
    # Main MW loop
    for t in 1:T
        # Get P_t from oracle
        P_temp, z_1, z_2 = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)
        
        # Update weights using exponential weights
        w_1star = w_1 * exp(eta * z_1)
        w_2star = w_2 * exp(eta * z_2)
        
        # Renormalize
        sum_w = w_1star + w_2star
        w_1 = w_1star / sum_w
        w_2 = w_2star / sum_w
        
        # Add to sum of P_t
        P .+= P_temp
        
        # Record progress
        P_average = (1/t) .* P
        loss_A_avg = (1/m_A) * (alpha - sum(covA .* P_average))
        loss_B_avg = (1/m_B) * (beta - sum(covB .* P_average))
        
        push!(record, [t, w_1, w_2, z_1, z_2, loss_A_avg, loss_B_avg])
    end
    
    # Take average of P_t
    P ./= T
    
    # Calculate loss of P_average
    z_1 = (1/m_A) * (alpha - sum(covA .* P))
    z_2 = (1/m_B) * (beta - sum(covB .* P))
    z = max(z_1, z_2)
    
    # Get last iterate (need to call oracle again or store from last iteration)
    # We'll store P_last from the last iteration
    P_last = P_temp
    
    # Calculate loss of P_last
    zl_1 = (1/m_A) * (alpha - sum(covA .* P_last))
    zl_2 = (1/m_B) * (beta - sum(covB .* P_last))
    z_last = max(zl_1, zl_2)
    
    println("MW method is finished.")
    println("Loss for group A: $z_1, for group B: $z_2")
    
    # Print record
    for row in record
        println(row)
    end
    
    return P, z, P_last, z_last
end

function oracle(n::Int, A::Matrix{Float64}, m_A::Int, B::Matrix{Float64}, m_B::Int, 
              alpha::Float64, beta::Float64, d::Int, w_1::Float64, w_2::Float64)
    """
    Oracle function: solves the weighted PCA problem
    """
    
    # Check sizes
    if size(A) != (m_A, n) || size(B) != (m_B, n)
        error("Input matrix to oracle method has wrong size.")
        return zeros(n, n), 0.0, 0.0
    end
    
    # Compute covariance matrices
    covA = A' * A
    covB = B' * B
    
    # Combine weighted data
    # Note: sqrt(w_1/m_A) scales group A, sqrt(w_2/m_B) scales group B
    combined_data = vcat(sqrt(w_1/m_A) .* A, sqrt(w_2/m_B) .* B)
    
    # Perform PCA (SVD) on combined data
    # In Julia, we use svd for PCA
    F = svd(combined_data)
    
    # Get top d principal components
    if d <= size(F.Vt, 2)
        # Extract first d right singular vectors
        V_d = F.Vt[1:d, :]'
        # Projection matrix P = V_d * V_d'
        P_o = V_d * V_d'
    else
        # If d is larger than available dimensions, use all
        P_o = Matrix{Float64}(I, n, n)
    end
    
    # Calculate losses
    z_1 = (1/m_A) * (alpha - sum(covA .* P_o))
    z_2 = (1/m_B) * (beta - sum(covB .* P_o))
    
    return P_o, z_1, z_2
end

function optApprox(X::Matrix{Float64}, d::Int)
    """
    Compute optimal d-dimensional approximation of X using SVD
    Returns the approximation matrix
    """
    m, n = size(X)
    
    if d >= min(m, n)
        return X  # Full rank, no approximation needed
    end
    
    # Compute SVD
    F = svd(X)
    
    # Keep top d singular values and vectors
    U_d = F.U[:, 1:d]
    Σ_d = Diagonal(F.S[1:d])
    V_d = F.V[:, 1:d]
    
    # Reconstruct approximation
    X_approx = U_d * Σ_d * V_d'
    
    return X_approx
end

# Optional: A more efficient version that avoids repeated covariance computations
function mw_efficient(A::Matrix{Float64}, B::Matrix{Float64}, d::Int, eta::Float64, T::Int)
    """
    More efficient version that precomputes covariance matrices
    """
    println("Efficient MW method is called")
    
    # Precompute covariance matrices
    covA = A' * A
    covB = B' * B
    
    # Get sizes
    m_A, n = size(A)
    m_B = size(B, 1)
    
    # Compute optimal d-dimensional approximations using SVD
    F_A = svd(A)
    Ahat = F_A.U[:, 1:d] * Diagonal(F_A.S[1:d]) * F_A.V[:, 1:d]'
    alpha = norm(Ahat)^2
    
    F_B = svd(B)
    Bhat = F_B.U[:, 1:d] * Diagonal(F_B.S[1:d]) * F_B.V[:, 1:d]'
    beta = norm(Bhat)^2
    
    # MW initialization
    w_1 = 0.5
    w_2 = 0.5
    
    P = zeros(n, n)
    record = Vector{Vector{Any}}()
    push!(record, ["iteration", "w_1", "w_2", "loss A", "loss B"])
    
    for t in 1:T
        # Call oracle with precomputed covariance matrices
        P_temp, z_1, z_2 = oracle_efficient(covA, covB, m_A, m_B, alpha, beta, n, d, w_1, w_2)
        
        # Update weights
        w_1star = w_1 * exp(eta * z_1)
        w_2star = w_2 * exp(eta * z_2)
        
        sum_w = w_1star + w_2star
        w_1 = w_1star / sum_w
        w_2 = w_2star / sum_w
        
        P .+= P_temp
        
        push!(record, [t, w_1, w_2, z_1, z_2])
    end
    
    P ./= T
    
    # Calculate final losses
    z_1 = (1/m_A) * (alpha - sum(covA .* P))
    z_2 = (1/m_B) * (beta - sum(covB .* P))
    z = max(z_1, z_2)
    
    println("Efficient MW method is finished.")
    println("Loss for group A: $z_1, for group B: $z_2")
    
    return P, z
end

function oracle_efficient(covA::Matrix{Float64}, covB::Matrix{Float64}, 
                       m_A::Int, m_B::Int, alpha::Float64, beta::Float64,
                       n::Int, d::Int, w_1::Float64, w_2::Float64)
    """
    Efficient oracle that works with precomputed covariance matrices
    """
    # The combined covariance matrix for weighted PCA
    # This is equivalent to (sqrt(w_1)*A; sqrt(w_2)*B)' * (sqrt(w_1)*A; sqrt(w_2)*B)
    # = w_1*A'*A + w_2*B'*B
    weighted_cov = (w_1/m_A) * covA + (w_2/m_B) * covB
    
    # Eigen decomposition for PCA
    # We want the top d eigenvectors of weighted_cov
    eigen_vals, eigen_vecs = eigen(weighted_cov)
    
    # Sort eigenvalues in descending order
    idx = sortperm(eigen_vals, rev=true)
    
    # Get top d eigenvectors
    if d <= n
        V_d = eigen_vecs[:, idx[1:d]]
        P_o = V_d * V_d'
    else
        P_o = Matrix{Float64}(I, n, n)
    end
    
    # Calculate losses
    z_1 = (1/m_A) * (alpha - sum(covA .* P_o))
    z_2 = (1/m_B) * (beta - sum(covB .* P_o))
    
    return P_o, z_1, z_2
end

# Example usage
function example_usage()
    # Generate example data
    A,B=quick_load_matrices()
    #M,A,B=Credit()
    # MW parameters
    eta = 0.1
    T = 50
    d=1
    # Run MW algorithm
    time=@elapsed begin
    P, z, P_last, z_last = mw(A, B, 1, eta, T)
    end
    println("\nResults:")
    println("Projection matrix size: $(size(P))")
    println("Rank: $(rank(P_last))")
    println("Maximum loss: $z")
    println("Time:$time")
    print(min(tr(A*P_last), tr(B*P_last)))
    return P, z, P_last, z_last
end

# Run example

example_usage()
