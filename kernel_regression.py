import numpy as np

def kernel_regression_poly(X, y, deg = 2):
    """
    This function calculates the kernel matrix for the given input data `X` 
    using a polynomial kernel of degree `deg` and solves for the regression 
    coefficients (alpha) using the pseudo-inverse.

    Parameters:
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the training input data.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) representing the target values for training.
    deg : int, optional
        The degree of the polynomial kernel. Default is 2.

    Returns:
    --------
    alpha : numpy.ndarray
        The regression coefficients obtained by solving the kernel ridge regression 
        problem.
    kernel_matrix : numpy.ndarray
        The computed kernel matrix based on the polynomial kernel for the training data. 
        Shape is (n_samples, n_samples).

    Notes:
    ------
    The polynomial kernel used here is defined as:
        K(x, x') = (x @ x' + 1)^deg
    where `x` and `x'` are input vectors and `deg` is the degree of the polynomial.
    """
    kernel_matrix = np.power(X @ X.T + 1, deg)
    alpha = np.matmul(np.linalg.pinv(kernel_matrix), y)
    
    return alpha, kernel_matrix


def kernel_pred_poly(alpha, X, X_test, deg = 2):
    """
    This function uses the learned regression coefficients `alpha` from 
    `kernel_regression_poly` and makes predictions on the test data `X_test`
    using a polynomial kernel of degree `deg`.

    Parameters:
    -----------
    alpha : numpy.ndarray
        The learned regression coefficients from the kernel regression, shape (n_samples,).
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the training input data.
    X_test : numpy.ndarray
        A 2D array of shape (n_test_samples, n_features) representing the test input data 
        for which predictions are to be made.
    deg : int, optional
        The degree of the polynomial kernel. Default is 2.

    Returns:
    --------
    preds : numpy.ndarray
        The predicted target values for the test data, shape (n_test_samples,).
    
    Notes:
    ------
    The polynomial kernel used for prediction is defined as:
        K(x, x') = (x @ x' + 1)^deg
    where `x` is a training vector and `x'` is a test vector.

    """
    pred_kernel_matrix = np.power(X @ X_test.T + 1, deg)     
    preds = np.sum(alpha * pred_kernel_matrix, axis = 0).T
    
    return preds