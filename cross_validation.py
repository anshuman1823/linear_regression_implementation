from gradient_descent import gradient_descent
from stochastic_gd import stochastic_gd
from gradient_descent_ridge import gradient_descent_ridge
from mean_squared_error import mse
import numpy as np

def cross_validation(model, X, y, k, max_iter = 10**4, step_size_multiplier = 0.01, C = None, batch = None):
    """
    Perform k-fold cross-validation on the given dataset using the specified model.

    This function divides the dataset into `k` folds and trains the model `k` times, each time 
    using a different fold as the validation set and the remaining `k-1` folds as the training set.
    It returns the average mean-squared error (MSE) across all folds, as well as the list of MSE 
    values for each fold.

    Parameters:
    -----------
    model : str
        Specifies the model to use for training. Should be one of:
        - "gd" : Gradient Descent for linear regression
        - "sgd" : Stochastic Gradient Descent for linear regression
        - "ridge" : Gradient Descent for Ridge regression
    X : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the input data, where each row 
        corresponds to a data point and each column corresponds to a feature.
    y : numpy.ndarray
        A 1D array of shape (n_samples,) containing the target values for the input data.
    k : int
        The number of folds for k-fold cross-validation.
    max_iter : int, optional
        Maximum number of iterations for the model's optimization algorithm. Default is 10^4.
    step_size_multiplier : float, optional
        Multiplier for the step size, which controls the rate of convergence. Default is 0.01.
    C : float, optional
        Regularization constant for Ridge regression. This parameter is required if `model="ridge"`.
    batch : int, optional
        Batch size for Stochastic Gradient Descent (SGD). This parameter is required if `model="sgd"`.

    Returns:
    --------
    mean_mse : float
        The average mean-squared error (MSE) across all folds.
    mse_list : list of floats
        A list containing the MSE for each fold in the k-fold cross-validation.

    Notes:
    ------
    - If `model` is "ridge", the `C` parameter (regularization coefficient) must be specified.
    - If `model` is "sgd", the `batch` parameter (batch size) must be specified.
    - The function uses the `gradient_descent`, `stochastic_gd`, and `gradient_descent_ridge` functions 
      from imported modules for the respective models.

    """

    model_dict = {"gd" : gradient_descent,
    "sgd" : stochastic_gd,
    "ridge": gradient_descent_ridge}
    
    if model not in model_dict.keys():
        print("model parameter not correctly passed. Please specify a model from (gd) gradient descent, (sgd) stochastic gradient descent, and (ridge) ridge regression.")
        return
    elif model == "ridge" and C is None:
        print("also pass the C parameter for the ridge model")
        return
    elif model == "sgd" and batch is None:
        print("also pass the batch parameter for the sgd model")
        return
    else:
        model_fun = model_dict[model]
    
    ## Setting a random seed
    np.random.seed(23)
    sample_perm = np.random.permutation(np.arange(0, len(X)))
    
    l = len(X)//k  ## Specifies the number of sample points in each fold

    X_batches = []  ## list will contain training batches
    y_batches = []
    val_batches = []  ## list will contain validation batches
    y_val_batches = []    

    for i in range(k):
        sample_fold = sample_perm[i*l:(i+1)*l]
        val_batches.append(X[sample_fold])
        y_val_batches.append(y[sample_fold])
        X_batches.append(X[np.setdiff1d(sample_perm, sample_fold)])
        y_batches.append(y[np.setdiff1d(sample_perm, sample_fold)])    
    
    mse_list = []

    for i in range(k):
        if model == "ridge":
            w = model_fun(X = X_batches[i], y = y_batches[i], C = C, max_iter = max_iter, step_size_multiplier = step_size_multiplier)
        elif model == "sgd":
            w = model_fun(X = X_batches[i], y = y_batches[i], batch = batch, max_iter = max_iter, step_size_multiplier = step_size_multiplier)
        else:
            w = model_fun(X = X_batches[i], y = y_batches[i], max_iter = max_iter, step_size_multiplier = step_size_multiplier)

        preds = np.matmul(val_batches[i], w)
        mse_list.append(mse(y_val_batches[i], preds))
    
    return np.mean(mse_list), mse_list