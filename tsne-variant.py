"""
Enhanced t-SNE Implementation - Simplified Version
===================================================

This is a simplified, procedural version of eh-tsne.py that removes OOP complexity.
It implements three optimization methods for t-SNE:
1. Gradient Descent (GD)
2. Momentum Method (MM)
3. Nesterov Accelerated Gradient (NAG)

Additionally, it provides both iterative and continuous (ODE-based) solution paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, iv


# ============================================================================
# PART 1: BASIC UTILITIES (Same as simple t-SNE)
# ============================================================================

def compute_pairwise_distances(X):
    """
    Calculate squared Euclidean distances between all pairs of points.
    Uses efficient matrix operations: ||a - b||² = ||a||² + ||b||² - 2·a·b
    
    INPUT:
        X: shape (n_samples, n_features)
    
    OUTPUT:
        D: shape (n_samples, n_samples) - squared distance matrix
    """
    # Sum of squared values for each sample
    sum_X = np.sum(np.square(X), axis=1)
    
    # Efficient computation of squared distances
    D = -2 * np.dot(X, X.T) + sum_X[:, np.newaxis] + sum_X[np.newaxis, :]
    
    # Ensure no negative values due to numerical precision
    D = np.maximum(D, 0)
    
    return D


def compute_entropy_and_probabilities(D, beta):
    """
    Convert distances to probabilities using Gaussian kernel.
    Calculate Shannon entropy to measure effective neighborhood size.
    
    INPUT:
        D: 1D array of squared distances from one point to others
        beta: precision parameter (inverse of variance)
    
    OUTPUT:
        H: entropy value
        P: probability distribution (sums to 1)
        sumP: sum before normalization (for diagnostics)
    """
    # Apply Gaussian kernel
    P = np.exp(-D * beta)
    
    # Sum of probabilities
    sumP = np.sum(P)
    
    # Handle edge case
    if sumP == 0:
        sumP = 1e-12
    
    # Normalize to probability distribution
    P = P / sumP
    
    # Calculate Shannon entropy
    H = np.log(sumP) + beta * np.sum(D * P)
    
    return H, P, sumP


def compute_joint_probabilities(X, perplexity=30.0, tol=1e-5, verbose=False):
    """
    Compute high-dimensional similarity matrix P using perplexity.
    Uses binary search to find appropriate beta (precision) for each point.
    
    INPUT:
        X: shape (n_samples, n_features)
        perplexity: effective number of neighbors (typically 5-50)
        tol: tolerance for binary search
        verbose: print progress
    
    OUTPUT:
        P: shape (n_samples, n_samples) - joint probability matrix
        beta: shape (n_samples,) - precision parameters for each point
    """
    print("Computing joint probabilities...")
    n = X.shape[0]
    
    # Calculate pairwise distances
    D = compute_pairwise_distances(X)
    
    # Initialize
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(perplexity)
    
    # For each data point
    for i in range(n):
        if verbose and i % 100 == 0:
            print(f"Processing point {i}/{n}")
        
        # Binary search bounds
        betamin = -np.inf
        betamax = np.inf
        
        # Get distances to all other points (excluding self)
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        
        # Calculate initial entropy
        H, thisP, sumP = compute_entropy_and_probabilities(Di, beta[i])
        Hdiff = H - logU
        tries = 0
        
        # Binary search for beta that gives desired perplexity
        while np.abs(Hdiff) > tol and tries < 100:
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 1.2
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 1.2
                else:
                    beta[i] = (beta[i] + betamin) / 2.0
            
            # Recalculate entropy
            H, thisP, sumP = compute_entropy_and_probabilities(Di, beta[i])
            Hdiff = H - logU
            tries += 1
            
            # Early stopping to prevent divergence
            if sumP < 1e-12 or Hdiff > 1e3:
                if verbose:
                    print(f"Early stopping at point {i}, try {tries}")
                break
        
        # Store probabilities for this point
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
    # Make P symmetric (joint probability)
    P = (P + P.T) / (2 * n)
    
    print(f"Mean value of sigma: {np.mean(np.sqrt(1 / beta)):.4f}")
    
    return P, beta


# ============================================================================
# PART 2: OPTIMIZATION METHODS (GD, MM, NAG)
# ============================================================================

def compute_Q_matrix(Y):
    """
    Compute low-dimensional similarity matrix Q using Student-t distribution.
    
    INPUT:
        Y: shape (n_samples, 2) - current 2D positions
    
    OUTPUT:
        Q: shape (n_samples, n_samples) - similarity matrix in low dimensions
        num: numerator values (needed for gradient calculation)
    """
    n = Y.shape[0]
    
    # Calculate pairwise squared distances in 2D
    sum_Y = np.sum(np.square(Y), 1)
    num = 1 / (1 + (-2 * np.dot(Y, Y.T) + sum_Y[:, None] + sum_Y[None, :]))
    
    # Set diagonal to 0
    np.fill_diagonal(num, 0)
    
    # Normalize to get Q
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    
    return Q, num


def compute_gradient(P, Q, Y, num):
    """
    Compute gradient of KL divergence for updating Y positions.
    
    INPUT:
        P: high-dimensional similarities
        Q: low-dimensional similarities
        Y: current 2D positions
        num: numerator from Q calculation
    
    OUTPUT:
        dY: gradient, shape (n_samples, 2)
    """
    n = Y.shape[0]
    dY = np.zeros_like(Y)
    
    PQ = P - Q
    
    for i in range(n):
        dY[i] = np.sum(np.tile(PQ[:, i] * num[:, i], (2, 1)).T * (Y[i] - Y), 0)
    
    return dY


def calculate_kl_divergence(P, Q):
    """
    Calculate KL divergence between P and Q.
    This measures how well the low-dimensional representation preserves similarities.
    
    INPUT:
        P: target similarities (high-dimensional)
        Q: current similarities (low-dimensional)
    
    OUTPUT:
        cost: KL divergence value
    """
    # Only calculate where both P and Q are non-zero
    mask = (P > 0) & (Q > 0)
    cost = np.sum(P[mask] * np.log(P[mask] / Q[mask]))
    return cost


def tsne_gradient_descent(X, P, max_iter=1000, h=0.1, alpha=4.0, verbose=True):
    """
    Standard Gradient Descent optimization for t-SNE.
    
    INPUT:
        X: shape (n_samples, n_features) - preprocessed data
        P: joint probability matrix
        max_iter: number of iterations
        h: step size (learning rate)
        alpha: early exaggeration factor (multiplied by P in early iterations)
    
    OUTPUT:
        Y_history: dict with Y positions at each iteration
        cost_history: dict with KL divergence at each iteration
    """
    print("Running Gradient Descent optimization...")
    n = X.shape[0]
    
    # Initialize Y randomly
    Y = np.random.randn(n, 2) * 0.0001
    
    # Storage
    Y_history = {0: Y.copy()}
    cost_history = {}
    
    # Apply early exaggeration
    P_exag = P * alpha
    
    for iter in range(max_iter):
        # Use exaggerated P for early iterations
        P_current = P_exag if iter < 250 else P
        
        # Compute Q and gradient
        Q, num = compute_Q_matrix(Y)
        dY = compute_gradient(P_current, Q, Y, num)
        
        # Update Y with gradient descent
        Y = Y + h * dY
        
        # Store
        Y_history[iter + 1] = Y.copy()
        
        # Calculate cost
        if iter % 10 == 0:
            cost = calculate_kl_divergence(P_current, Q)
            cost_history[iter] = cost
            if verbose:
                print(f"Iteration {iter}: KL divergence = {cost:.4f}")
    
    return Y_history, cost_history


def tsne_momentum_method(X, P, max_iter=1000, h=0.1, momentum=0.5, alpha=4.0, verbose=True):
    """
    Momentum Method (MM) optimization for t-SNE.
    Uses momentum to accelerate convergence and smooth updates.
    
    INPUT:
        X: shape (n_samples, n_features)
        P: joint probability matrix
        max_iter: number of iterations
        h: step size
        momentum: momentum coefficient (typically 0.5-0.9)
        alpha: early exaggeration factor
    
    OUTPUT:
        Y_history: dict with Y positions at each iteration
        cost_history: dict with KL divergence at each iteration
    """
    print("Running Momentum Method optimization...")
    n = X.shape[0]
    
    # Initialize
    Y = np.random.randn(n, 2) * 0.0001
    Y_prev = Y.copy()
    
    # Storage
    Y_history = {-1: Y.copy(), 0: Y.copy()}
    cost_history = {}
    
    # Apply early exaggeration
    P_exag = P * alpha
    
    for iter in range(max_iter):
        # Use exaggerated P for early iterations
        P_current = P_exag if iter < 250 else P
        
        # Compute Q and gradient
        Q, num = compute_Q_matrix(Y)
        dY = compute_gradient(P_current, Q, Y, num)
        
        # Update Y with momentum
        Y_new = Y + h * dY + momentum * (Y - Y_prev)
        
        # Update for next iteration
        Y_prev = Y.copy()
        Y = Y_new
        
        # Store
        Y_history[iter + 1] = Y.copy()
        
        # Calculate cost
        if iter % 10 == 0:
            cost = calculate_kl_divergence(P_current, Q)
            cost_history[iter] = cost
            if verbose:
                print(f"Iteration {iter}: KL divergence = {cost:.4f}")
    
    return Y_history, cost_history


def tsne_nesterov_method(X, P, max_iter=1000, h=0.1, alpha=4.0, verbose=True):
    """
    Nesterov Accelerated Gradient (NAG) optimization for t-SNE.
    Uses Nesterov momentum for faster convergence.
    
    INPUT:
        X: shape (n_samples, n_features)
        P: joint probability matrix
        max_iter: number of iterations
        h: step size
        alpha: early exaggeration factor
    
    OUTPUT:
        Y_history: dict with Y positions at each iteration
        cost_history: dict with KL divergence at each iteration
    """
    print("Running Nesterov Accelerated Gradient optimization...")
    n = X.shape[0]
    
    # Initialize
    Y = np.random.randn(n, 2) * 0.0001
    Y_nes = Y.copy()  # Nesterov's extrapolated position
    
    # Storage
    Y_history = {0: Y.copy()}
    cost_history = {}
    
    # Apply early exaggeration
    P_exag = P * alpha
    
    for iter in range(max_iter):
        # Use exaggerated P for early iterations
        P_current = P_exag if iter < 250 else P
        
        # Compute Q and gradient at Nesterov's position
        Q, num = compute_Q_matrix(Y_nes)
        dY = compute_gradient(P_current, Q, Y_nes, num)
        
        # Update Y
        Y_new = Y_nes + h * dY
        
        # Nesterov momentum update
        Y_nes_new = Y_new + (iter / (iter + 3)) * (Y_new - Y)
        
        # Update for next iteration
        Y = Y_new
        Y_nes = Y_nes_new
        
        # Store
        Y_history[iter + 1] = Y.copy()
        
        # Calculate cost
        if iter % 10 == 0:
            cost = calculate_kl_divergence(P_current, Q)
            cost_history[iter] = cost
            if verbose:
                print(f"Iteration {iter}: KL divergence = {cost:.4f}")
    
    return Y_history, cost_history


# ============================================================================
# PART 3: ODE-BASED SOLUTION PATH (Continuous/Analytical Approach)
# ============================================================================

def compute_eigendecomposition(P, alpha=4.0):
    """
    Compute eigendecomposition of Laplacian matrix.
    This enables analytical (ODE-based) solution paths.
    
    L = D - P, where D is degree matrix
    Then solve for modified Laplacian: L(alpha * P - H_n)
    
    INPUT:
        P: joint probability matrix
        alpha: early exaggeration coefficient
    
    OUTPUT:
        eigenvalues: sorted eigenvalues
        eigenvectors: corresponding eigenvectors
    """
    print("Computing eigendecomposition for ODE solution...")
    n = P.shape[0]
    
    # Compute Laplacian
    D = np.diag(P.sum(axis=1))
    L = D - P
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort by eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Modify eigenvalues according to the formula
    for i in range(len(eigenvalues)):
        if i == 0:
            eigenvalues[i] = alpha * eigenvalues[i]
        else:
            eigenvalues[i] = (alpha * eigenvalues[i]) - 1 / (n - 1)
    
    return eigenvalues, eigenvectors


def bessel_term(t, eigenvalue):
    """
    Calculate Bessel function term for ODE solution.
    Uses standard Bessel function (jv) for positive eigenvalues,
    modified Bessel function (iv) for negative eigenvalues.
    
    INPUT:
        t: time parameter (continuous version of iteration * step_size)
        eigenvalue: eigenvalue from Laplacian
    
    OUTPUT:
        Bessel function value
    """
    if t * eigenvalue == 0:
        return 1
    elif eigenvalue > 0:
        # Standard Bessel function of first kind, order 1
        return (2 / (t * np.sqrt(eigenvalue))) * jv(1, t * np.sqrt(eigenvalue))
    else:
        # Modified Bessel function of first kind, order 1
        return (2 / (t * np.sqrt(-eigenvalue))) * iv(1, t * np.sqrt(-eigenvalue))


def ode_solution_gradient_descent(Y0, eigenvalues, eigenvectors, k, h):
    """
    Compute ODE solution for Gradient Descent at iteration k.
    Continuous/analytical solution: Y(t) = exp(-t * Lambda) * Y0
    
    INPUT:
        Y0: initial positions
        eigenvalues: from Laplacian decomposition
        eigenvectors: from Laplacian decomposition
        k: iteration number
        h: step size
    
    OUTPUT:
        Y: positions at iteration k
    """
    t = k * h
    n = Y0.shape[0]
    Y = np.zeros((n, 2))
    
    # Compute solution using eigendecomposition
    for dim in range(2):  # x and y coordinates
        for l in range(len(eigenvalues)):
            coeff = np.dot(eigenvectors[:, l], Y0[:, dim])
            Y[:, dim] += np.exp(-t * eigenvalues[l]) * coeff * eigenvectors[:, l]
    
    return Y


def ode_solution_momentum_method(Y0, eigenvalues, eigenvectors, k, h, momentum):
    """
    Compute ODE solution for Momentum Method at iteration k.
    Uses modified eigenvalues accounting for momentum.
    
    INPUT:
        Y0: initial positions
        eigenvalues: from Laplacian decomposition
        eigenvectors: from Laplacian decomposition
        k: iteration number
        h: step size
        momentum: momentum coefficient
    
    OUTPUT:
        Y: positions at iteration k
    """
    t = k * h
    n = Y0.shape[0]
    Y = np.zeros((n, 2))
    
    # Momentum modifies the effective eigenvalues
    for dim in range(2):
        for l in range(len(eigenvalues)):
            coeff = np.dot(eigenvectors[:, l], Y0[:, dim])
            effective_eigenval = eigenvalues[l] / (1 - momentum)
            Y[:, dim] += np.exp(-t * effective_eigenval) * coeff * eigenvectors[:, l]
    
    return Y


def ode_solution_nesterov_method(Y0, eigenvalues, eigenvectors, k, h):
    """
    Compute ODE solution for Nesterov Method at iteration k.
    Uses Bessel functions to model Nesterov acceleration analytically.
    
    INPUT:
        Y0: initial positions
        eigenvalues: from Laplacian decomposition
        eigenvectors: from Laplacian decomposition
        k: iteration number
        h: step size
    
    OUTPUT:
        Y: positions at iteration k
    """
    t = k * np.sqrt(h)  # Note: time scales differently for NAG
    n = Y0.shape[0]
    Y = np.zeros((n, 2))
    
    # NAG solution uses Bessel functions
    for l in range(len(eigenvalues)):
        bessel_val = bessel_term(t, eigenvalues[l])
        for dim in range(2):
            Y[:, dim] += Y0[l, dim] * bessel_val * eigenvectors[:, l]
    
    return Y


# ============================================================================
# PART 4: EVALUATION METRICS
# ============================================================================

def simple_pca(X, n_components=50):
    """
    Simple PCA implementation for dimensionality reduction.
    
    INPUT:
        X: shape (n_samples, n_features) - data matrix
        n_components: number of principal components to keep
    
    OUTPUT:
        X_reduced: shape (n_samples, n_components) - transformed data
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(-eigenvalues.real)
    eigenvectors = eigenvectors[:, idx]
    
    # Keep top n_components
    eigenvectors = eigenvectors[:, :n_components]
    
    # Project data
    X_reduced = np.dot(X_centered, eigenvectors.real)
    
    return X_reduced


def simple_kmeans(X, n_clusters, max_iter=100, random_state=42):
    """
    Simple k-means clustering implementation.
    
    INPUT:
        X: shape (n_samples, n_features) - data to cluster
        n_clusters: number of clusters
        max_iter: maximum number of iterations
        random_state: random seed for reproducibility
    
    OUTPUT:
        labels: shape (n_samples,) - cluster assignment for each sample
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Initialize centroids randomly from data points
    idx = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[idx].copy()
    
    for _ in range(max_iter):
        # Assign points to nearest centroid
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                new_centroids[i] = np.mean(X[labels == i], axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels


def calculate_ari(Y, y_true):
    """
    Calculate Adjusted Rand Index using k-means clustering.
    Measures how well the embedding preserves cluster structure.
    
    ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    where RI is the Rand Index
    
    INPUT:
        Y: embedded coordinates, shape (n_samples, 2)
        y_true: true labels
    
    OUTPUT:
        ARI score (higher is better, 1.0 is perfect)
    """
    n_clusters = len(np.unique(y_true))
    
    # K-means clustering on embedded coordinates
    clusters = simple_kmeans(Y, n_clusters=n_clusters, random_state=42)
    
    # Compute contingency table
    n = len(y_true)
    contingency = np.zeros((n_clusters, n_clusters), dtype=int)
    
    for true_label in np.unique(y_true):
        for pred_label in range(n_clusters):
            contingency[int(true_label), pred_label] = np.sum(
                (y_true == true_label) & (clusters == pred_label)
            )
    
    # Sum over rows and columns
    sum_comb_c = np.sum([n_c * (n_c - 1) / 2 for n_c in np.sum(contingency, axis=1)])
    sum_comb_k = np.sum([n_k * (n_k - 1) / 2 for n_k in np.sum(contingency, axis=0)])
    sum_comb = np.sum([n_ij * (n_ij - 1) / 2 for n_ij in contingency.flatten()])
    
    # Expected index (for adjusted Rand index)
    expected_index = sum_comb_c * sum_comb_k / (n * (n - 1) / 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    # Adjusted Rand Index
    if max_index - expected_index == 0:
        return 0.0
    
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    
    return ari


# ============================================================================
# PART 5: MAIN WORKFLOW & COMPARISON
# ============================================================================

def compare_optimization_methods(X, y, perplexity=30.0, max_iter=500, h=0.1):
    """
    Compare all three optimization methods (GD, MM, NAG) on the same dataset.
    
    INPUT:
        X: data, shape (n_samples, n_features)
        y: labels for evaluation
        perplexity: t-SNE perplexity parameter
        max_iter: iterations for each method
        h: step size
    
    OUTPUT:
        results: dict with Y_history and cost_history for each method
    """
    print("\n" + "="*70)
    print("COMPARING OPTIMIZATION METHODS")
    print("="*70 + "\n")
    
    # Preprocess with PCA
    print("Preprocessing with PCA...")
    X_reduced = simple_pca(X, n_components=50)
    
    # Compute joint probabilities (same for all methods)
    P, beta = compute_joint_probabilities(X_reduced, perplexity=perplexity)
    
    # Run each method
    results = {}
    
    print("\n" + "-"*70)
    Y_hist_gd, cost_hist_gd = tsne_gradient_descent(
        X_reduced, P, max_iter=max_iter, h=h, verbose=False
    )
    results['GD'] = {'Y_history': Y_hist_gd, 'cost_history': cost_hist_gd}
    
    print("\n" + "-"*70)
    Y_hist_mm, cost_hist_mm = tsne_momentum_method(
        X_reduced, P, max_iter=max_iter, h=h, momentum=0.5, verbose=False
    )
    results['MM'] = {'Y_history': Y_hist_mm, 'cost_history': cost_hist_mm}
    
    print("\n" + "-"*70)
    Y_hist_nag, cost_hist_nag = tsne_nesterov_method(
        X_reduced, P, max_iter=max_iter, h=h, verbose=False
    )
    results['NAG'] = {'Y_history': Y_hist_nag, 'cost_history': cost_hist_nag}
    
    return results


def visualize_comparison(results, y, title="Optimization Method Comparison"):
    """
    Visualize the final embeddings from each optimization method.
    
    INPUT:
        results: dict from compare_optimization_methods
        y: labels for coloring
        title: plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = ['GD', 'MM', 'NAG']
    method_names = {
        'GD': 'Gradient Descent',
        'MM': 'Momentum Method',
        'NAG': 'Nesterov Accelerated Gradient'
    }
    
    for idx, method in enumerate(methods):
        Y_final = results[method]['Y_history'][max(results[method]['Y_history'].keys())]
        
        ax = axes[idx]
        scatter = ax.scatter(Y_final[:, 0], Y_final[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
        ax.set_title(method_names[method])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    plt.colorbar(scatter, ax=axes, label='Digit Label')
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cost_convergence(results, max_iter_plot=500):
    """
    Plot KL divergence over iterations for all methods.
    
    INPUT:
        results: dict from compare_optimization_methods
        max_iter_plot: max iteration to plot
    """
    plt.figure(figsize=(10, 6))
    
    methods = ['GD', 'MM', 'NAG']
    colors = {'GD': 'red', 'MM': 'blue', 'NAG': 'green'}
    method_names = {
        'GD': 'Gradient Descent',
        'MM': 'Momentum Method',
        'NAG': 'Nesterov Accelerated Gradient'
    }
    
    for method in methods:
        cost_hist = results[method]['cost_history']
        iters = sorted([k for k in cost_hist.keys() if k <= max_iter_plot])
        costs = [cost_hist[k] for k in iters]
        
        plt.plot(iters, costs, label=method_names[method], 
                color=colors[method], linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('KL Divergence', fontsize=12)
    plt.title('Convergence Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def load_digits_data():
    """
    Load digits dataset from sklearn (if available) or generate sample data.
    
    OUTPUT:
        X: shape (n_samples, 64) - digit images as 8x8 flattened
        y: shape (n_samples,) - digit labels 0-9
    """
    try:
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        return X, y
    except ImportError:
        print("sklearn not available. Generating sample data...")
        # Generate synthetic data with 10 clusters
        np.random.seed(42)
        n_samples_per_class = 180
        n_classes = 10
        n_features = 64
        
        X = []
        y = []
        
        for i in range(n_classes):
            # Create cluster center
            center = np.random.randn(n_features) * 10
            # Generate samples around center
            samples = center + np.random.randn(n_samples_per_class, n_features) * 2
            X.append(samples)
            y.append(np.full(n_samples_per_class, i))
        
        X = np.vstack(X)
        y = np.hstack(y)
        
        # Normalize to [0, 16] range like digits dataset
        X = (X - X.min()) / (X.max() - X.min()) * 16
        
        return X, y


if __name__ == "__main__":
    """
    Example: Compare optimization methods on digits dataset.
    
    WORKFLOW:
    1. Load digits dataset (1797 samples, 64 features)
    2. Run all three optimization methods (GD, MM, NAG)
    3. Visualize final embeddings
    4. Plot convergence curves
    """
    
    print("\n" + "="*70)
    print("ENHANCED t-SNE - SIMPLIFIED VERSION")
    print("Comparing Gradient Descent, Momentum, and Nesterov methods")
    print("="*70 + "\n")
    
    # Load data
    print("Loading digits dataset...")
    X, y = load_digits_data()
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y)}\n")
    
    # Compare methods
    results = compare_optimization_methods(
        X, y, 
        perplexity=30.0, 
        max_iter=500, 
        h=0.1
    )
    
    # Visualize
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70 + "\n")
    
    visualize_comparison(results, y)
    plot_cost_convergence(results, max_iter_plot=490)
    
    # Calculate ARI for each method
    print("\nAdjusted Rand Index (ARI) for final embeddings:")
    for method in ['GD', 'MM', 'NAG']:
        Y_final = results[method]['Y_history'][max(results[method]['Y_history'].keys())]
        ari = calculate_ari(Y_final, y)
        print(f"  {method}: {ari:.4f}")
    
    print("\nDone!")
