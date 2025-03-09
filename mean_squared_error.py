import numpy as np

def mse(true, preds):
        """
        Calculate the Mean Squared Error (MSE) between the actual values and the predicted values.
        
        Parameters:
        -----------
        true : numpy.ndarray
                The actual values of the target variable (ground truth). It is a 1D array.
        preds : numpy.ndarray
                The predicted values of the target variable. It is a 1D array.

        Returns:
        --------
        mean_squared_error : float
                The mean squared error, which is the average of the squared differences between the actual 
                and predicted values.
        """
        return np.mean(np.power(true - preds, 2), axis = 0)