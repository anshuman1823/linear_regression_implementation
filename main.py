## importing necessary libraries and functions

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

## loading the algorithms from their modules

from gradient_descent import gradient_descent
from stochastic_gd import stochastic_gd
from least_squares_solution import least_squares_soln
from gradient_descent_ridge import gradient_descent_ridge
from cross_validation import cross_validation
from mean_squared_error import mse
from kernel_regression import kernel_regression_poly, kernel_pred_poly

## loading the dataset

dataset_f = os.path.join(os.curdir, "datasets")
train_df = pd.read_csv(os.path.join(dataset_f, "FMLA1Q1Data_train.csv"), header = None)
test_df = pd.read_csv(os.path.join(dataset_f, "FMLA1Q1Data_test.csv"), header = None)

X = np.array(train_df.loc[:, :1])
y = np.array(train_df.loc[:, 2:])

X_test = np.array(test_df.loc[:, :1])
y_test = np.array(test_df.loc[:, 2:])

## Adding a bias term to the X dataset by adding a column on 1's at 0 column index

X = np.c_[np.ones(len(X)), X]
X_test = np.c_[np.ones(len(X_test)), X_test]


# ------------------Task-1-----------------------------------------------------------------------------------------------------

## using the least_squares_soln to get w_ml

w_ml = least_squares_soln(X, y)
print(f"w_ml: {w_ml}")

## Evaluating the w_ml solution on the test dataset

preds = np.matmul(X_test, w_ml)
mse_lss = mse(y_test, preds)
print(f"MSE for OLS equation: {mse_lss}")


# ------------------Task-2-----------------------------------------------------------------------------------------------------

## Finding the correct step_size for the gradient descent function

fig, ax = plt.subplots(nrows=3, ncols=2, figsize = (12,8))

for p, axis in zip([-1, -2, -2.5, -3, -3.5, -4], ax.ravel()):
    w_gd, w_t = gradient_descent(X, y, step_size_multiplier=10**p, intermediate_w=True)
    
    preds = np.matmul(X_test, w_gd)
    mse_gd = mse(y_test, preds)
    
    ## Plotting a figure showing values w took during gradient descent. x-axis: w co-efficient for feature_0, and y_axis: w co-efficient for feature_1. 
    axis.scatter(w_t[:,1], w_t[:,2], label = "wt obtained from gradient descent at t iteration", zorder = 1)
    axis.plot(w_t[:,1], w_t[:,2], linestyle = "--", linewidth = 1, zorder = 1)
    axis.scatter(w_ml[1], w_ml[2], label = "w_ml obtained from least squares solution", color = "red", s = 30, zorder = 2)
    axis.annotate("w_0", (w_t[0,1], w_t[0,2]), textcoords="offset points", xytext=(80, 10), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder = 3)
    axis.annotate("w_" + str(len(w_t) - 1), (w_t[-1,1], w_t[-1,2]), textcoords="offset points", xytext=(-100, 10), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder = 3)
    axis.set_xlabel("w co-efficient for Feature 0")
    axis.set_ylabel("w co-efficient for Feature 1")
    axis.set_title(f"step_size_multiplier = 10**{p}")
    axis.legend()
    
plt.suptitle("Plotting wt obtained from gradient descent at t iteration vs w_ml obtained from least squares solution", y = 0.99, size = 15)
plt.tight_layout()
plt.show()

## Using the gradient descent algorithm to solve for w_gd

w_gd, w_t = gradient_descent(X, y, step_size_multiplier=10**-3.5, intermediate_w=True)
print(f"w_gd: {w_gd}")

## Evaluating the w_gd solution on the test dataset

preds = np.matmul(X_test, w_gd)
mse_gd = mse(y_test, preds)
print(f" MSE for linear regression using gradient descent: {mse_gd}")


## Calculating the norm of the difference of w_ml and w_t at t iteration

diff = w_t - w_ml
norm_diff = np.linalg.norm(diff, axis=1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15,8), sharex = True)

ax[0].plot(np.arange(len(norm_diff)), norm_diff, label = "||w_t − w_ml||")
ax[0].set_xlabel("t: iteration")
ax[0].set_ylabel("L-2 norm : ||w_t - w_ml||")
ax[0].legend()
ax[1].plot(np.arange(len(norm_diff)), np.log(norm_diff), label = " log(||w_t − w_ml||)")
ax[1].set_xlabel("t: iteration")
ax[1].set_ylabel("log of L-2 norm : log(||w_t - w_ml||)")
ax[1].legend()
plt.suptitle("Plotting L-2 norm ||w_t − w_ml|| as a function of t for gradient descent")
plt.show()


# ------------------Task-3-----------------------------------------------------------------------------------------------------

## Using the stochastic gradient descent algorithm to solve for w_gd

w_sgd, w_t = stochastic_gd(X, y, step_size_multiplier=10**-2, batch = 100, intermediate_w=True, max_iter=10**4)
print(f"w_sgd: {w_sgd}")

## Evaluating the w_sgd solution on the test dataset

preds = np.matmul(X_test, w_sgd)
mse_sgd = mse(y_test, preds)
print(f"MSE obtained from stochastic gradient algorithm: {mse_sgd}")


## Plotting a figure showing values w took during stochastic gradient descent. x-axis: w co-efficient for feature_0, and y_axis: w co-efficient for feature_1.

fig = plt.figure(figsize = (12, 8))
plt.scatter(w_t[:,1], w_t[:,2], label = "wt obtained from stochastic gradient descent at successive t iterations", alpha = 0.2, zorder = 1)
plt.plot(w_t[:,1], w_t[:,2], linestyle = "--", linewidth = 1, zorder = 1)
plt.annotate("w_0", (w_t[0,1], w_t[0,2]), textcoords="offset points", xytext=(10, 10), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder = 3)
plt.annotate("w_" + str(len(w_t) - 1), (w_t[-1,1], w_t[-1,2]), textcoords="offset points", xytext=(-100, 10), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder = 3)
plt.scatter(w_ml[1], w_ml[2], label = "w_ml obtained from least squares solution", color = "red", s = 30, zorder = 2)
plt.xlabel("w co-efficient for Feature 0")
plt.ylabel("w co-efficient for Feature 1")
plt.title("Plotting wt obtained from stochastic gradient descent at t iteration vs w_ml obtained from least squares solution")
plt.legend()
plt.show()

## Calculating the norm of the difference of w_ml and w_t at t iteration for stochastic gradient descent

diff = w_t - w_ml
norm_diff = np.linalg.norm(diff, axis=1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15,8), sharex = True)

ax[0].plot(np.arange(len(norm_diff)), norm_diff, label = "||w_t − w_ml||")
ax[0].set_xlabel("t: iteration")
ax[0].set_ylabel("L-2 norm : ||w_t - w_ml||")
ax[0].legend()

ax[1].plot(np.arange(len(norm_diff)), np.log(norm_diff), label = "log(||w_t − w_ml||)")
ax[1].set_xlabel("t: iteration")
ax[1].set_ylabel("log of L-2 norm : log(||w_t - w_ml||)")
ax[1].legend()

plt.suptitle("Plotting L-2 norm ||w_t − w_ml|| as a function of t for stochastic gradient descent")
plt.show()


# ------------------Task-4-----------------------------------------------------------------------------------------------------

## Creating an array containing possible values for c
c_values = np.arange(10**-1, 10, 10**-1)

mse_ridge_list = []  ## mse for each value of c obtained from cross validation will get stored in this list

for c in c_values:
    mse_ridge, _ = cross_validation("ridge", X, y, k = 5,  C = c, step_size_multiplier=10**-2)
    mse_ridge_list.append(mse_ridge)

## Finding the value of regularization parameter c_opt (equivalent to lambda given in the question) obtained after cross-validation
c_opt = np.round(c_values[np.argmin(mse_ridge_list)],3)

print(f" Lowest MSE of {np.min(mse_ridge)} obtained at C = {np.round(c_opt, 3)}")

## Plotting C (ridge regularization parameter) vs validation MSE from 5-fold cross-validation

fig = plt.figure(figsize = (8,8))
plt.scatter(c_values, mse_ridge_list, label = "MSE")
plt.axvline(c_opt, linestyle = "--", color = "red", label = "Minimum MSE")
plt.annotate(f"c_opt: {np.round(c_opt,3)}", (c_opt, np.min(mse_ridge_list)), textcoords="offset points", xytext=(60, 100), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
plt.xlabel("C: Ridge Regularization parameter")
plt.ylabel("MSE on validation dataset from k-fold cross validation")
plt.title("C (ridge regularization parameter) vs validation MSE from 5-fold cross-validation")
plt.legend()
plt.show()

## Fitting ridge regression model using c_opt

w_ridge, _ = gradient_descent_ridge(X, y, C = c_opt, step_size_multiplier=10**-3.5, intermediate_w=True)
print(f" w obtained from ridge linear regression with c = {c_opt} is w: {w_ridge}")

## Evaluating the w_sgd solution on the test dataset

preds = np.matmul(X_test, w_ridge)
mse_ridge = mse(y_test, preds)
print(f"MSE obtained from ridge gradient algorithm: {mse_ridge}")

## The mse obtained on the test set after regularization is slightly lower than the mse obtained from least squares solution or gradient descent solution

# ------------------Task-5-----------------------------------------------------------------------------------------------------

# Analysing the train dataset to choose the best kernel
# Plotting feature-0 vs feature-1 with contours based on the target variable value

fig, ax = plt.subplots(figsize=(12, 8))

x_min, x_max = train_df.loc[:, 0].min(), train_df.loc[:, 0].max()
y_min, y_max = train_df.loc[:, 1].min(), train_df.loc[:, 1].max()
xi, yi = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)) # Creating a grid for contour plotting

zi = griddata((train_df.loc[:, 0], train_df.loc[:, 1]), train_df.loc[:, 2], (xi, yi), method='cubic') ## Interpolating the target values on the grid and plotting the contours
contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.5)

scatter = ax.scatter(train_df.loc[:, 0], train_df.loc[:, 1], c=train_df.loc[:, 2], s=train_df.loc[:, 2]*4, cmap='viridis', edgecolors='black', linewidth=0.5)

plt.colorbar(scatter, label='Target variable ealue')

ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.set_title("(Size is propotional to y)")
plt.suptitle("Feature-0 vs Feature-1 with Contours Based on target variable value", size = 18, y= .95)

plt.show()  # From this figure, we can see that the target variable value is more as we move away from the origin in the third quadrant and the first quadrant. The label value seems to be dependent on the magnitude of both f_0 and f_1. When both f_0 and f_1 have high magnitude, the label value is high. When both the f_0 and f_1 value are having low magnitudes, the feature value is low. Also, when one of the features is having high value and the other is having low value, then also the label has low value. This suggests that the label is related to the term f_0 * f_1.

## Analyzing the correlation of the features with the output target variable value

fig, ax = plt.subplots(nrows = 1, ncols=2, figsize = (15,7), sharey=True)
f_0 = train_df.loc[:, 0]
f_1 = train_df.loc[:, 1]
y_ = train_df.loc[:, 2]
ax[0].scatter(f_0, y_)
ax[0].set_title("Feature_0 vs target variable y")
ax[0].set_xlabel("Feature_0")
ax[0].set_ylabel("Target Variable")
ax[1].scatter(f_1, y_)
ax[1].set_title("Feature_1 vs target variable y")
ax[1].set_xlabel("Feature_1")
ax[1].set_ylabel("Target Variable")
plt.show()

## Setting the degree for the polynomial kernel regression
deg = 2

## Using the kernel_regression_poly function to get the alpha values
alpha, km = kernel_regression_poly(X, y, deg = deg)

## Making predictions on the test dataset using the obtained alpha values
preds = kernel_pred_poly(alpha, X, X_test, deg = deg)
mse_kernel = mse(y_test.ravel(), preds.ravel())
print(f" MSE for kernel regression using polynomial kernel with degree = 2: {mse_kernel}") 

# here we can see that we are getting a significant reduction in the mse value as compared to the mse of 66 
# which we obtained from the linear regression model. hence, polynomial kernel with degree 2 is a good choice for the dataset.