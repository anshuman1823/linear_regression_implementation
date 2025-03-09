import numpy as np

def gradient_descent(X, y, max_iter = 10**4, step_size_multiplier = 0.01, intermediate_w = False):
    """
    Perform gradient descent to solve linear regression.

    This function implements the gradient descent algorithm to minimize the squared loss
    for a linear regression problem. It iteratively updates the weight vector `w` using
    the gradient of the loss function with respect to `w`, until convergence or the maximum
    number of iterations is reached.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the input data, where each row 
        corresponds to a data point and each column corresponds to a feature.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) containing the target values for the input data.
    max_iter : int, optional
        Maximum number of iterations for the gradient descent algorithm. Default is 10^4.
    step_size_multiplier : float, optional
        Multiplier for the step size, which controls the rate of convergence. Default is 0.01.
    intermediate_w : bool, optional
        If True, returns a list of the weight vectors `w` at each iteration. Default is False.

    Returns:
    --------
    w : numpy.ndarray
        The optimized weight vector `w` of shape (n_features, 1) obtained after running 
        gradient descent.
    w_array : numpy.ndarray, optional
        If `intermediate_w` is True, returns a 2D array containing the weight vectors `w` 
        at each iteration. Shape is (iterations, n_features).

    Notes:
    ------
    - The gradient of the loss function is computed as:
        ∇L(w) = 2 * X^T (X*w - y)
    - Gradient descent stops when the norm of the gradient (∇L(w)) becomes smaller than 
      10^(-6), indicating convergence. If convergence is reached, the number of iterations 
      is printed.
    - The step size decreases as the number of iterations increases, with a step size at 
      iteration `i` being `1 / (i + 1)`.

    """
    
    def del_fx(X, y, w):
        """
        Returns the value of the derivative of the loss function for a particular w
        """
        return 2*(X.T @ X @ w) - 2*(X.T @ y)

    iters = 0
    step = 1
    np.random.seed(23)
    w = np.random.randn(X.T.shape[0], 1)
    w_array = [w]
    
    while (iters < max_iter) and np.linalg.norm(del_fx(X, y, w)) > 10**-6:
        w = w - step * step_size_multiplier * del_fx(X, y, w)
        w_array.append(w)
        iters += 1
        step = 1/(iters + 1)

    if iters < max_iter and not np.isnan(np.linalg.norm(w)):
        print(f"Gradient descent converged with {iters} iterations")
        
    if intermediate_w:
        return w, np.array(w_array)
    
    return w
        

