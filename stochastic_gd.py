import numpy as np

def stochastic_gd(X, y, batch = 100, max_iter = 10**4, step_size_multiplier = 0.01, intermediate_w = False):
    """
    Perform stochastic gradient descent (SGD) for linear regression.

    This function implements the stochastic gradient descent algorithm, where a subset (batch) 
    of the data is randomly selected at each iteration to compute the gradient, rather than 
    using the entire dataset. This can lead to faster convergence for large datasets.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the input data, where each row 
        corresponds to a data point and each column corresponds to a feature.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) containing the target values for the input data.
    batch : int, optional
        The number of data points to be sampled at each iteration to compute the gradient. Default is 100.
    max_iter : int, optional
        Maximum number of iterations for the stochastic gradient descent algorithm. Default is 10^4.
    step_size_multiplier : float, optional
        Multiplier for the step size, which controls the rate of convergence. Default is 0.01.
    intermediate_w : bool, optional
        If True, returns a 2D array of the weight vectors `w` at each iteration. Default is False.

    Returns:
    --------
    w : numpy.ndarray
        The final weight vector `w`, averaged over all iterations. Shape is (n_features, 1).
    w_array : numpy.ndarray, optional
        If `intermediate_w` is True, returns a 2D array containing the weight vectors `w` at each iteration.
        Shape is (iterations, n_features).

    Notes:
    ------
    - At each iteration, a random batch of data points is sampled to compute the gradient. 
    - The gradient of the loss function is computed as:
        ∇L(w) = 2 * X_batch^T (X_batch*w - y_batch)
    - The algorithm will stop after a maximum number of iterations or when convergence is manually 
      checked using the returned weight vectors.
    - The step size decreases as the number of iterations increases, with the step size at 
      iteration `i` being `1 / (i + 1)`.
    """
    
    def del_fx(X, y, w):
        """
        Returns the value of the derivative of the loss function for a particular w
        """
        return 2*((X.T @ X) @ w) - 2*(X.T @ y)
    
    iters = 0
    step = 1
    np.random.seed(69)
    w = np.random.randn(X.T.shape[0], 1)
    w_array = [w]
    
    while (iters < max_iter):
        points = np.random.randint(0, len(X), batch)   ## Selecting random samples from the dataset
        X_ = X[points]
        y_ = y[points]
        w = w - step * step_size_multiplier * del_fx(X_, y_, w)
        w_array.append(w)
        iters += 1
        step = 1/(iters + 1)

    w_array = np.array(w_array)
    
    if intermediate_w:
        return np.mean(w_array, axis = 0), w_array
    
    return np.mean(w_array, axis = 0)
        

