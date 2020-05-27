"""This module contains the regression tests."""
import numba
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal
from scipy.optimize.optimize import fminbound

from robupy.get_worst_case import criterion_full
from robupy.get_worst_case import get_worst_case_probs
from robupy.minimize_scalar import fminbound_numba
from robupy.tests.get_random_testcase import get_request
from robupy.tests.resources.pre_numba.auxiliary import pre_numba_criterion_full
from robupy.tests.resources.pre_numba.auxiliary import pre_numba_get_worst_case_probs


def test_2():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        get_worst_case_probs(v, q, beta, is_cost),
        pre_numba_get_worst_case_probs(v, q, beta, is_cost),
    )


def test_3():
    y = np.random.random() * np.random.randint(100)
    lower = y - np.random.random() * np.random.randint(100)
    upper = y + np.random.random() * np.random.randint(100)
    assert_allclose(
        fminbound_numba(f_fminbound, lower, upper),
        fminbound(f_fminbound, lower, upper, full_output=True),
    )


def test_4():
    x, v, q, beta, gamma, is_cost = get_request()
    # if is_cost:
    #     v_max_min = np.min(v)
    # else:
    #     v_max_min = np.max(v)
    assert_array_almost_equal(
        criterion_full(gamma, v, q, beta), pre_numba_criterion_full(v, q, beta, gamma),
    )


@numba.jit(nopython=True)
def f_fminbound(x):
    return x ** 2.0
