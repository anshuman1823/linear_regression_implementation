import numpy as np

def least_squares_soln(X, y):
    """
    This function calculates the optimal weight vector `w*` for linear regression
    using the least squares.

    Parameters:
    -----------
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the input data, 
        where each row corresponds to a data point and each column corresponds to a feature.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) containing the target values for the input data.

    Returns:
    --------
    numpy.ndarray
        The optimal weight vector `w*` that minimizes the least squares error, shape 
        will be (n_features,).
    
    Notes:
    ------
    The least squares solution is obtained by solving the normal equation:
        w* = (X^T X)^(-1) X^T y
    
    """

    lhs = np.linalg.pinv(X.T @ X)  ## Here, we use the pseudo-inverse to handle cases where X^T X is not invertible.
    rhs = X.T @ y
    return lhs @ rhs