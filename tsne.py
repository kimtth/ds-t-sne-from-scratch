from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) Implementation from Scratch
================================================================================

PURPOSE: Reduce high-dimensional data (e.g., 64 dimensions) to 2D/3D for visualization
while preserving local neighborhood relationships.

END-TO-END FLOW OVERVIEW:
1. Start with high-dimensional data (e.g., 1000 samples × 64 features)
2. Calculate how similar each point is to every other point in high dimensions
3. Create a random 2D map of these points
4. Iteratively adjust the 2D positions to match the high-dimensional similarities
5. Output: 2D coordinates that can be plotted
"""

# ------------------------------------------------------------
# Utility: Compute pairwise squared Euclidean distances
# ------------------------------------------------------------
def pairwise_distances(X):
    """
    Calculate the distance between every pair of data points.
    
    INPUT:
        X: 2D array of shape (n_samples, n_features)
           Example: 1000 images, each with 64 pixel values
           Shape: (1000, 64)
    
    OUTPUT:
        D: 2D array of shape (n_samples, n_samples)
           D[i, j] = squared distance between point i and point j
           Shape: (1000, 1000)
           Example values: D[0, 1] = 234.5 (distance from point 0 to point 1)
    
    HOW IT WORKS:
        - For each pair of points (i, j), calculates ||point_i - point_j||²
        - Uses matrix operations to compute all distances at once (fast)
        - Result is a symmetric matrix where diagonal is zero (distance to self = 0)
    
    SAMPLE DATA FLOW:
        Input:  X = [[1.2, 0.5, ...], [2.1, 0.8, ...], ...] → shape (1000, 50)
        Output: D = [[0, 45.2, 123.7, ...], 
                     [45.2, 0, 89.1, ...],
                     [123.7, 89.1, 0, ...], ...] → shape (1000, 1000)
    """
    # sum_X[i] = sum of squared values in row i
    # Shape: (n_samples,)
    # Example: sum_X = [15.2, 18.9, 12.1, ...] for n samples
    sum_X = np.sum(np.square(X), 1)
    
    # Efficient computation: ||a - b||² = ||a||² + ||b||² - 2·a·b
    # Returns symmetric distance matrix
    return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]


# ------------------------------------------------------------
# Utility: Shannon entropy and conditional probability for a given beta
# ------------------------------------------------------------
def Hbeta(D, beta=1.0):
    """
    Convert distances to probabilities using a Gaussian (bell curve) and calculate entropy.
    
    INPUT:
        D: 1D array of squared distances from one point to all others
           Example: [0, 45.2, 123.7, 89.1, ...] for one data point
           Shape: (n_samples - 1,) — excludes distance to itself
        
        beta: Precision parameter (controls spread of the Gaussian)
              Higher beta = narrower distribution (only nearby points matter)
              Lower beta = wider distribution (distant points also matter)
    
    OUTPUT:
        H: Entropy (a measure of how "spread out" the probability distribution is)
           Single number, e.g., 4.91 (relates to perplexity)
        
        P: Probability distribution over neighbors
           1D array, shape: (n_samples - 1,)
           Values sum to 1.0, e.g., [0.25, 0.15, 0.10, ...]
           Interpretation: Probability that this point would pick each other point as a neighbor
    
    HOW IT WORKS:
        - Applies Gaussian formula: P ∝ exp(-distance² × beta)
        - Nearby points get high probability, distant points get low probability
        - Normalizes so all probabilities sum to 1
        - Calculates entropy to measure effective neighborhood size
    
    SAMPLE DATA FLOW:
        Input:  D = [45.2, 123.7, 89.1, 12.5, ...] → distances from 1 point
                beta = 0.5 → precision parameter
        Output: P = [0.18, 0.05, 0.08, 0.35, ...] → probabilities (sum=1.0)
                H = 4.91 → entropy value
    """
    # Apply Gaussian kernel: closer points get higher values
    # Example: D[0]=45.2, beta=0.5 → P[0]=exp(-45.2*0.5)=exp(-22.6)≈0.00000016
    #          D[3]=12.5, beta=0.5 → P[3]=exp(-12.5*0.5)=exp(-6.25)≈0.0019 (closer!)
    P = np.exp(-D * beta)
    
    # Set self-probability to 0 (a point is not its own neighbor)
    P[np.arange(len(D)), np.arange(len(D))] = 0
    
    sumP = np.sum(P)
    if sumP == 0:
        # Edge case: all probabilities are zero
        H = 0
        P = np.zeros_like(D)
    else:
        # Normalize to create proper probability distribution (sums to 1)
        P = P / sumP
        
        # Calculate Shannon entropy: measures "effective number of neighbors"
        H = np.log(sumP) + beta * np.sum(D * P)
    
    return H, P


# ------------------------------------------------------------
# Step 1: Compute high-dimensional similarities (P matrix)
# ------------------------------------------------------------
def x2p(X, tol=1e-5, perplexity=30.0):
    """
    For each data point, find how similar it is to every other point in high dimensions.
    
    INPUT:
        X: High-dimensional data after PCA preprocessing
           Shape: (n_samples, n_features_reduced)
           Example: (1000, 50) — 1000 data points with 50 features each
        
        perplexity: Controls the effective number of neighbors to consider
                    Typical values: 5-50 (30 is common default)
                    Higher perplexity = considers more neighbors
        
        tol: Tolerance for binary search convergence (1e-5 is very precise)
    
    OUTPUT:
        P: Similarity matrix in high-dimensional space
           Shape: (n_samples, n_samples)
           Example shape: (1000, 1000)
           P[i, j] = probability that point i would pick point j as a neighbor
           Values are between 0 and 1, rows sum to ~1
           Example row: [0, 0.05, 0.12, 0.03, 0.08, ...] (1000 values)
    
    HOW IT WORKS:
        - For each point, uses binary search to find the right "neighborhood size"
        - The neighborhood size is adjusted until we have approximately 'perplexity' effective neighbors
        - Creates a probability distribution where nearby points have high probability
    
    SAMPLE DATA FLOW:
        Input:  X = [[1.2, 0.5, ..., 0.8],   ← 1000 samples
                     [2.1, 0.8, ..., 1.2],
                     ...] → shape (1000, 50)
                perplexity = 30.0
        
        Process: For point 0: [0.05, 0.12, 0.03, ...] → row 0 of P
                 For point 1: [0.04, 0.08, 0.15, ...] → row 1 of P
                 ...
        
        Output: P = [[0.00, 0.05, 0.12, 0.03, ...],  ← point 0's similarities
                     [0.04, 0.00, 0.08, 0.15, ...],  ← point 1's similarities
                     [...]] → shape (1000, 1000)
    """
    print("Computing pairwise distances...")
    (n, d) = X.shape  # n = number of samples, d = number of features
    
    # Calculate all pairwise distances (n × n matrix)
    D = pairwise_distances(X)
    
    # Initialize probability matrix (will be filled row by row)
    P = np.zeros((n, n))
    
    # Initialize beta (precision parameter) for each point
    beta = np.ones((n, 1))
    
    # Target entropy value (log of perplexity)
    logU = np.log(perplexity)

    # Process each data point to find its similarity to all others
    # LOOP INPUT: Each iteration processes one point (i) out of n total points
    for i in range(n):
        if i % 100 == 0:
            print(f"Computing P-values for point {i} of {n}...")

        # Binary search bounds for beta (will be adjusted)
        betamin, betamax = -np.inf, np.inf
        
        # Get distances from point i to all other points (excluding itself)
        # Di shape: (n-1,) — one distance value to each other point
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        
        # Calculate initial probabilities and entropy with current beta
        H, thisP = Hbeta(Di, beta[i])
        Hdiff = H - logU  # Difference between actual and target entropy
        tries = 0

        # Binary search to find beta that gives us the desired perplexity
        # GOAL: Adjust beta until entropy matches log(perplexity)
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                # Entropy too high → distribution too spread out → increase beta (narrow it)
                betamin = beta[i].copy()
                beta[i] = 2.0 * beta[i] if np.isinf(betamax) else (beta[i] + betamax) / 2.0
            else:
                # Entropy too low → distribution too concentrated → decrease beta (widen it)
                betamax = beta[i].copy()
                beta[i] = beta[i] / 2.0 if np.isinf(betamin) else (beta[i] + betamin) / 2.0

            # Recalculate with new beta
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Store the computed probabilities for this point (row i of P matrix)
        # thisP shape: (n-1,) — probability distribution over all other points
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Report average sigma (standard deviation of Gaussian kernels)
    print("Mean value of sigma:", np.mean(np.sqrt(1 / beta)))
    
    # FUNCTION OUTPUT: P matrix (n × n) with high-dimensional similarities
    return P


# ------------------------------------------------------------
# Step 2–5: Main t-SNE function
# ------------------------------------------------------------
def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000, lr=200.0):
    """
    Main t-SNE algorithm: Reduce high-dimensional data to 2D/3D for visualization.
    
    ============================================================================
    INPUT:
        X: Original high-dimensional data
           Shape: (n_samples, n_features)
           Example: (1797, 64) — 1797 digit images, each 8×8 pixels = 64 features
        
        no_dims: Target dimensionality (usually 2 for visualization)
        
        initial_dims: Reduce to this many dimensions using PCA before t-SNE
                      Speeds up computation and reduces noise
                      Default: 50
        
        perplexity: Controls neighborhood size (5-50 typical)
                    Think of it as "number of close neighbors to focus on"
        
        max_iter: Number of optimization iterations
                  More iterations = better results but slower
        
        lr: Learning rate for gradient descent (step size for updates)
    
    OUTPUT:
        Y: Low-dimensional embedding
           Shape: (n_samples, no_dims)
           Example: (1797, 2) — 1797 points with (x, y) coordinates
           These are the final 2D positions that can be plotted!
    
    ============================================================================
    ALGORITHM STEPS:
      1. PCA: Reduce dimensions from high (e.g., 64) to moderate (e.g., 50)
      2. Compute P matrix: High-dimensional similarities between all points
      3. Initialize Y: Random starting positions in 2D
      4. Optimization loop:
         - Compute Q matrix: Low-dimensional similarities
         - Calculate gradient: How to move points to better match P and Q
         - Update Y: Move points in the direction that reduces mismatch
      5. Return final Y: Optimized 2D positions
    ============================================================================
    """
    
    # ========================================================================
    # STEP 1: PCA PREPROCESSING
    # ========================================================================
    # PURPOSE: Reduce very high dimensions (e.g., 784 for MNIST) to moderate (e.g., 50)
    # WHY: Makes t-SNE faster and filters out noise
    
    # Center the data: subtract mean from each feature
    # Before: X might have values [0, 255] for pixels
    # After: X centered around 0, e.g., [-127, 128]
    X = X - np.mean(X, axis=0)
    
    # Compute covariance matrix: measures how features vary together
    # cov shape: (n_features, n_features), e.g., (64, 64)
    cov = np.cov(X.T)
    
    # Find principal components (directions of maximum variance)
    # eig_vals: importance of each component
    # eig_vecs: directions of each component
    eig_vals, eig_vecs = np.linalg.eig(cov)
    
    # Sort components by importance (largest eigenvalues first)
    idx = np.argsort(-eig_vals.real)
    
    # Keep only the top 'initial_dims' components
    # eig_vecs shape: (n_features, initial_dims), e.g., (64, 50)
    eig_vecs = eig_vecs[:, idx[:initial_dims]]
    
    # Project data onto principal components
    # X shape changes: (1797, 64) → (1797, 50)
    # NOW: X has 1797 samples with 50 features each (reduced from 64)
    X = np.dot(X, eig_vecs.real)
    
    # SAMPLE DATA AFTER PCA:
    # Before: X[0] = [0, 0, 5, 13, 9, ...] (64 pixel values, range 0-16)
    # After:  X[0] = [1.2, -0.8, 3.1, ...] (50 principal components, centered)

    # ========================================================================
    # STEP 2: COMPUTE HIGH-DIMENSIONAL SIMILARITIES (P matrix)
    # ========================================================================
    # PURPOSE: Calculate how similar each point is to every other in high dimensions
    # This is our TARGET — we want to preserve these relationships in 2D
    
    # Compute conditional probabilities using perplexity-based Gaussian
    # P shape: (n_samples, n_samples), e.g., (1797, 1797)
    # P[i, j] = similarity between point i and point j
    P = x2p(X, 1e-5, perplexity)
    
    # Make P symmetric: P[i,j] = P[j,i] for all i, j
    # Original P is conditional: P(j|i) — probability of j given i
    # Symmetric P: joint probability P(i,j)
    # Normalize so entire matrix sums to 1
    P = (P + P.T) / (2 * np.sum(P))
    
    # Prevent numerical issues (avoid log(0) later)
    P = np.maximum(P, 1e-12)
    
    # NOW: P matrix represents TARGET similarities in high dimensions
    # Example: P[0, 1] = 0.023 means points 0 and 1 are somewhat similar
    
    # SAMPLE P MATRIX (showing first 5×5 corner):
    # P = [[0.0000, 0.0234, 0.0012, 0.0456, 0.0089],
    #      [0.0234, 0.0000, 0.0567, 0.0023, 0.0145],
    #      [0.0012, 0.0567, 0.0000, 0.0234, 0.0678],
    #      [0.0456, 0.0023, 0.0234, 0.0000, 0.0345],
    #      [0.0089, 0.0145, 0.0678, 0.0345, 0.0000]]
    # Note: Symmetric, diagonal is 0, values sum to 1.0 across entire matrix

    # ========================================================================
    # STEP 3: INITIALIZE LOW-DIMENSIONAL MAP (Y)
    # ========================================================================
    # PURPOSE: Create random starting positions in 2D space
    
    (n, d) = X.shape  # n = 1797 samples, d = 50 features (after PCA)
    
    # Y: The 2D coordinates we're trying to find
    # Shape: (n_samples, no_dims), e.g., (1797, 2)
    # Initialized randomly with small values (Gaussian distribution)
    # Example initial Y[0] = [0.23, -0.45] — random (x, y) position
    Y = np.random.randn(n, no_dims)
    
    # dY: Gradient (direction to move points)
    # Shape: (1797, 2)
    dY = np.zeros((n, no_dims))
    
    # iY: Momentum term (accumulated velocity for smoother updates)
    # Shape: (1797, 2)
    iY = np.zeros((n, no_dims))
    
    # Momentum: fraction of previous update to keep (helps smooth optimization)
    momentum = 0.5
    
    # SAMPLE INITIAL Y (first 5 points):
    # Y = [[ 0.234, -0.456],  ← point 0 at random position
    #      [-0.123,  0.789],  ← point 1
    #      [ 0.567, -0.234],  ← point 2
    #      [-0.890,  0.123],  ← point 3
    #      [ 0.345,  0.678]]  ← point 4
    # All random, will be optimized over iterations

    # ========================================================================
    # STEP 4: OPTIMIZATION LOOP (Gradient Descent)
    # ========================================================================
    # PURPOSE: Iteratively adjust Y positions to match high-dimensional similarities
    # GOAL: Make Q (low-dim similarities) match P (high-dim similarities)
    
    for iter in range(max_iter):
        # --------------------------------------------------------------------
        # STEP 4a: Compute Q matrix (low-dimensional similarities)
        # --------------------------------------------------------------------
        # Q represents current similarities in 2D based on current Y positions
        
        # Calculate pairwise squared distances in 2D
        # sum_Y[i] = x_i² + y_i² for each point
        sum_Y = np.sum(np.square(Y), 1)
        
        # Compute Student-t kernel: 1 / (1 + distance²)
        # num shape: (n_samples, n_samples), e.g., (1797, 1797)
        # num[i,j] = 1 / (1 + ||Y_i - Y_j||²)
        # Student-t gives heavier tails than Gaussian (helps avoid crowding)
        num = 1 / (1 + (-2 * np.dot(Y, Y.T) + sum_Y[:, None] + sum_Y[None, :]))
        
        # Set diagonal to 0 (a point is not similar to itself)
        np.fill_diagonal(num, 0)
        
        # Normalize to get probabilities (Q matrix)
        # Q shape: (1797, 1797)
        # Q[i,j] = current similarity between points i and j in 2D
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)  # Avoid division by zero
        
        # NOW: 
        # P = what similarities SHOULD be (from high-dimensional data)
        # Q = what similarities CURRENTLY are (from 2D positions in Y)
        
        # SAMPLE Q MATRIX at iteration 100 (first 5×5 corner):
        # Q = [[0.0000, 0.0189, 0.0034, 0.0423, 0.0067],
        #      [0.0189, 0.0000, 0.0501, 0.0045, 0.0198],
        #      [0.0034, 0.0501, 0.0000, 0.0267, 0.0712],
        #      [0.0423, 0.0045, 0.0267, 0.0000, 0.0389],
        #      [0.0067, 0.0198, 0.0712, 0.0389, 0.0000]]
        # Compare with P above - getting closer as iterations progress!

        # --------------------------------------------------------------------
        # STEP 4b: Compute gradient (direction to move each point)
        # --------------------------------------------------------------------
        # Gradient tells us how to move each point to reduce mismatch between P and Q
        
        # PQ[i,j] = difference between target and current similarity
        # Positive: points should be closer in 2D
        # Negative: points should be farther in 2D
        PQ = P - Q
        
        # For each point i, compute gradient: sum of forces from all other points
        # dY[i] is a 2D vector showing which direction to move point i
        for i in range(n):
            # Weighted sum of (Y[i] - Y[j]) vectors
            # Points with large PQ pull harder
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        
        # dY shape: (1797, 2)
        # Example: dY[0] = [0.15, -0.23] means move point 0 right and down
        
        # SAMPLE GRADIENT (first 5 points at iteration 100):
        # dY = [[ 0.152, -0.234],  ← point 0: move right & down
        #       [-0.089,  0.167],  ← point 1: move left & up
        #       [ 0.234, -0.089],  ← point 2: move right & down
        #       [-0.178,  0.098],  ← point 3: move left & up
        #       [ 0.123,  0.267]]  ← point 4: move right & up

        # --------------------------------------------------------------------
        # STEP 4c: Update positions using momentum
        # --------------------------------------------------------------------
        # Momentum: keeps some velocity from previous iterations (smoother, faster convergence)
        
        # Update momentum: iY = momentum * old_velocity - learning_rate * gradient
        iY = momentum * iY - lr * dY
        
        # Update positions: Y = Y + velocity
        Y = Y + iY
        
        # Re-center the map (keeps it stable, prevents drift)
        Y = Y - np.mean(Y, axis=0)
        
        # NOW: Y has been updated! Points have moved to better positions.
        # After many iterations, Y will converge to a good 2D representation.
        
        # SAMPLE Y EVOLUTION (point 0 over iterations):
        # Iter 0:   Y[0] = [ 0.234, -0.456]  (random start)
        # Iter 100: Y[0] = [ 2.145,  1.234]  (moved significantly)
        # Iter 200: Y[0] = [ 3.567,  2.891]  (still moving)
        # Iter 500: Y[0] = [ 4.123,  3.456]  (converging)
        # Final:    Y[0] = [ 4.089,  3.501]  (stable position)

        # --------------------------------------------------------------------
        # STEP 4d: Monitor progress
        # --------------------------------------------------------------------
        # Every 100 iterations, compute and print the error (KL divergence)
        if (iter + 1) % 100 == 0:
            # KL divergence: measures how different Q is from P
            # Lower is better (0 = perfect match)
            C = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter + 1}: error = {C:.4f}")

        # After 250 iterations, increase momentum for faster convergence
        if iter == 250:
            momentum = 0.8

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    # Y: Final 2D coordinates
    # Shape: (n_samples, no_dims), e.g., (1797, 2)
    # Each row is an (x, y) coordinate that can be plotted
    # Example: Y[0] = [12.3, -5.7], Y[1] = [10.1, -4.9], ...
    # These preserve the high-dimensional neighborhood structure!
    
    # SAMPLE FINAL OUTPUT (first 10 points of digits dataset):
    # Y = [[  4.089,   3.501],  ← digit '0' image
    #      [  3.912,   4.123],  ← digit '0' image (close to above!)
    #      [-12.345,  -8.901],  ← digit '1' image (far from '0's)
    #      [-11.678,  -9.234],  ← digit '1' image (close to other '1')
    #      [  8.234, -15.678],  ← digit '2' image
    #      [  9.012, -14.890],  ← digit '2' image
    #      ...]
    # Notice: Same digits cluster together with similar coordinates!
    return Y


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    """
    EXAMPLE: Visualize handwritten digits in 2D
    
    THE FLOW:
    1. Load data: 1797 digit images (8×8 pixels = 64 features each)
    2. Run t-SNE: Reduce from 64D to 2D
    3. Plot: Each digit becomes a point colored by its label (0-9)
    
    EXPECTED RESULT:
    - Similar digits (all 0s, all 1s, etc.) cluster together
    - You can see the structure of the data in 2D!
    """

    # --------------------------------------------------------------------
    # INPUT: Load the digits dataset
    # --------------------------------------------------------------------
    # X: shape (1797, 64) — 1797 images, each with 64 pixel values
    # y: shape (1797,) — labels from 0 to 9 (which digit each image represents)
    X, y = load_digits(return_X_y=True)
    print(f"Input data shape: {X.shape}")
    print(f"Example: First image has {X.shape[1]} features (pixels)")

    # --------------------------------------------------------------------
    # PROCESSING: Run t-SNE algorithm
    # --------------------------------------------------------------------
    # INPUT to tsne: X with shape (1797, 64)
    # OUTPUT from tsne: Y with shape (1797, 2)
    Y = tsne(X, no_dims=2, perplexity=30.0, max_iter=500)
    
    print(f"\nOutput data shape: {Y.shape}")
    print(f"Each of {Y.shape[0]} points now has {Y.shape[1]} coordinates (x, y)")

    # --------------------------------------------------------------------
    # OUTPUT: Visualize the 2D embedding
    # --------------------------------------------------------------------
    # Y[:, 0] = x-coordinates for all points
    # Y[:, 1] = y-coordinates for all points
    # c=y = color each point by its digit label (0-9)
    plt.figure(figsize=(8, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=10)
    plt.title("t-SNE from scratch (Digits dataset)")
    plt.colorbar(label="Digit label (0-9)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()
