# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.

    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """

    losses = np.zeros((len(grid_w0), len(grid_w1)))
    # ***************************************************
    for i in range(len(grid_w0)):
        for j in range(len(grid_w1)):
            w = np.array([grid_w0[i],grid_w1[j]])
            losses[i][j] = compute_loss(y, tx, w)
    # ***************************************************
    #raise NotImplementedError
    return losses

# def grid_search(y, tx, grid_w0, grid_w1):
#     """Algorithm for grid search."""
#     # Create a meshgrid of w0 and w1 values
#     W0, W1 = np.meshgrid(grid_w0, grid_w1)
    
#     # Reshape meshgrid matrices into vectors for broadcasting
#     w0_flat = W0.flatten()
#     w1_flat = W1.flatten()
    
#     # Compute predictions and loss in a vectorized manner
#     tx_flat = np.repeat(tx, len(grid_w0) * len(grid_w1), axis=0)
#     predictions = tx_flat.dot(np.column_stack((w0_flat, w1_flat)).T)
#     e = np.repeat(y, len(grid_w0) * len(grid_w1)) - predictions
#     loss_flat = np.mean(np.square(e), axis=0) / 2
    
#     # Reshape the loss vector back into the grid shape
#     losses = loss_flat.reshape(len(grid_w0), len(grid_w1))
    
#     return losses

#ctrl+K puis crtl+C   et ctrl+K puis crtl+U pour enlever

