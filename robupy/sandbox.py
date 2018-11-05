"""This module contains functions that are not yet ready for prime time due to the need to add
further flexibility and further testing."""

from statsmodels.sandbox.distributions.extras import mvnormcdf
import numpy as np


def get_normal_probabilities(num_bins, mean, cov):
    """This function returns a grid of binned normal probabilities."""
    # Currently this only works for four-dimensional normal distribution.
    num_dims = 4

    q = np.tile(np.nan, [num_bins] * num_dims)

    grids = list()
    for i in range(num_dims):
        scale = np.sqrt(cov[i, i])
        lower, upper = -1.96 * scale, 1.96 *  scale

        grid = np.linspace(lower, upper, num_bins - 1, endpoint=True)
        grid = np.concatenate(([-np.inf], grid, [np.inf]), axis=0)
        grids += [grid]

    wv, xv, yv, zv = np.meshgrid(*grids, indexing='ij')

    for i in range(1, num_bins + 1):
        for j in range(1, num_bins + 1):
            for k in range(1, num_bins + 1):
                for l in range(1, num_bins + 1):

                    w_upper, w_lower = wv[i, j, k, l], wv[i - 1, j, k, l]
                    x_upper, x_lower = xv[i, j, k, l], xv[i, j - 1, k, l]
                    y_upper, y_lower = yv[i, j, k, l], yv[i, j, k - 1, l]
                    z_upper, z_lower = zv[i, j, k, l], zv[i, j, k, l - 1]

                    upper = [w_upper, x_upper, y_upper, z_upper]
                    lower = [w_lower, x_lower, y_lower, z_lower]

                    q[i - 1, j - 1, k - 1, l - 1] = mvnormcdf(upper, mean, cov, lower)

    # Getting started with some basic consistency checks.
    np.testing.assert_equal(np.all(q >= 0), True)
    np.testing.assert_equal(0.98 < np.sum(q) < 1.02, True)

    # Scaling output to ensure that probabilities sum to one.
    q = q / np.sum(q)

    return q, grid
