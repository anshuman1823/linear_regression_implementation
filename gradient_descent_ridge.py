import numpy as np

def gradient_descent_ridge(X, y, C = 1, max_iter = 10**4, step_size_multiplier = 0.01, intermediate_w = False):
    """
    Perform gradient descent for ridge regression (L2-regularized linear regression).

    This function implements the gradient descent algorithm to solve ridge regression, 
    a form of linear regression that includes an L2 regularization term to penalize large 
    weight values. It iteratively updates the weight vector `w` using the gradient of the 
    ridge regression loss function until convergence or the maximum number of iterations 
    is reached.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the input data, 
        where each row corresponds to a data point and each column corresponds to a feature.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) containing target values for the input data.
    C : float, optional
        The regularization constant (equivalent to lambda in ridge regression). This controls 
        the strength of the L2 regularization, with larger values enforcing greater regularization. 
        Default is 1.
    max_iter : int, optional
        Maximum number of iterations for the gradient descent algorithm. Default is 10^4.
    step_size_multiplier : float, optional
        Multiplier for the step size, which controls the rate of convergence. Default is 0.01.
    intermediate_w : bool, optional
        If True, returns a 2D array of the weight vectors `w` at each iteration. Default is False.

    Returns:
    --------
    w : numpy.ndarray
        The final weight vector `w` obtained after running gradient descent. Shape is (n_features, 1).
    w_array : numpy.ndarray, optional
        If `intermediate_w` is True, returns a 2D array containing the weight vectors `w` 
        at each iteration. Shape is (iterations, n_features).

    Notes:
    ------
    - Ridge regression adds an L2 regularization term to the standard linear regression 
      loss function to prevent overfitting. The loss function for ridge regression is:
        L(w) = ||Xw - y||^2 + C * ||w||^2
      where `C` is the regularization constant.
    - The gradient of the ridge regression loss function is computed as:
        ∇L(w) = 2 * X^T (Xw - y) + 2 * C * w
    - The gradient descent stops when the norm of the gradient becomes smaller than 10^(-6), 
      indicating convergence.
    - The step size decreases as the number of iterations increases, with the step size at 
      iteration `i` being `1 / (i + 1)`.
    """
    
    def del_fx(X, y, w, C):
        """
        Returns the value of the derivative of the loss function for a particular w
        """
        return 2*((X.T @ X) @ w) - 2*(X.T @ y) + 2*C*w
    
    iters = 0
    step = 1
    np.random.seed(23)
    w = np.random.randn(X.T.shape[0], 1)
    w_array = [w]
    
    while (iters < max_iter) and np.linalg.norm(del_fx(X, y, w, C)) > 10**-6:
        w = w - step * step_size_multiplier * del_fx(X, y, w, C)
        w_array.append(w)
        iters += 1
        step = 1/(iters + 1)

    # if iters < max_iter and not np.isnan(np.linalg.norm(w)):
    #     print(f"Ridge gradient descent converged with {iters} iterations")
    
    if intermediate_w:
        return w, np.array(w_array)
    
    return w
        

