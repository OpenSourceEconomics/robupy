"""This module contains all checks for the package to improve readability."""
from scipy.stats import entropy
import numpy as np

from robupy.tests.resources.pre_numba.config import SMALL_FLOAT


def pre_numba_checks(label, *args):
    """This function hosts all checks."""
    if label == 'criterion_full_in':
        v, q, beta, lambda_ = args
        np.testing.assert_equal(beta >= 0, True)
        np.testing.assert_equal(lambda_ >= 0, True)
        np.testing.assert_almost_equal(sum(q), 1)
        np.testing.assert_equal(np.all(q > 0), True)
        np.testing.assert_equal(np.all(np.isfinite(v)), True)
    elif label == 'criterion_full_out':
        rslt, = args
        # We do encounter the case that the criterion function is infinite. However, this is not
        # a problem, since it is only called in a minimization which discards this evaluation point.
        np.testing.assert_equal(np.isfinite(rslt) or rslt == np.inf, True)
    elif label == 'calculate_p_in':
        v, q, lambda_ = args
        np.testing.assert_equal(lambda_ >= 0, True)
        np.testing.assert_almost_equal(sum(q), 1)
        np.testing.assert_equal(np.all(q > 0), True)
        np.testing.assert_equal(np.all(np.isfinite(v)), True)
    elif label == 'calculate_p_out':
        p, = args
        np.testing.assert_equal(np.all(p >= 0.0), True)
        np.testing.assert_almost_equal(sum(p), 1.0)
    elif label in ['get_worst_case_in', 'get_worst_case_outcome_in']:
        v, q, beta = args
        np.testing.assert_equal(np.all(np.isfinite(v)), True)
        np.testing.assert_almost_equal(sum(q), 1)
        np.testing.assert_equal(np.all(q > 0), True)
        np.testing.assert_equal(beta >= 0.0, True)
    elif label == 'get_worst_case_out':
        p, q, beta, rslt = args
        np.testing.assert_almost_equal(sum(p), 1)
        np.testing.assert_equal(np.all(p >= 0.0), True)
        np.testing.assert_almost_equal(sum(q), 1)
        np.testing.assert_equal(np.all(q > 0), True)
        np.testing.assert_equal(beta >= 0.0, True)
        np.testing.assert_equal(rslt[2] == 0, True)
        np.testing.assert_equal(entropy(p, q) - beta < 0.0001, True)
    elif label == 'get_worst_case_outcome_out':
        v, q, beta, is_cost, rslt = args
        np.testing.assert_equal(min(v) - SMALL_FLOAT <= rslt <= max(v) + SMALL_FLOAT, True)
    else:
        raise NotImplementedError
