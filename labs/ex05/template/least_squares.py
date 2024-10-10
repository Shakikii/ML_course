# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: shape=(N, 1), target vector
        tx: shape=(N, d), input data (with intercept term)

    Returns:
        w: optimal weights
        mse: mean squared error
    """
    # Step 1: Compute the optimal weights using the normal equation
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    # Step 2: Calculate the mean squared error (MSE)
    e = y - tx @ w  # residuals
    mse = (1 / (2 * y.shape[0])) * np.sum(e**2)
    
    return w, mse

