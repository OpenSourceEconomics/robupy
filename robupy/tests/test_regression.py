"""This module contains the regression tests."""

from numpy.testing import assert_array_almost_equal, assert_allclose
import numpy as np
import numba
from scipy.optimize.optimize import fminbound
from robupy.minimize import fminbound_numba
from robupy.auxiliary import get_worst_case_outcome, get_worst_case_probs
from robupy.tests.pre_numba.auxiliary import (
    pre_numba_get_worst_case_outcome,
    pre_numba_get_worst_case_probs,
)
from robupy.tests.auxiliary import get_request


def test_1():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        pre_numba_get_worst_case_outcome(v, q, beta, is_cost),
        get_worst_case_outcome(v, q, beta, is_cost),
    )


def test_2():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        get_worst_case_probs(v, q, beta, is_cost),
        pre_numba_get_worst_case_probs(v, q, beta, is_cost),
    )


def test_3():
    y = np.random.random() * np.random.randint(1)
    lower = y - np.random.random() * np.random.randint(1)
    upper = y + np.random.random() * np.random.randint(1)
    assert_allclose(
        fminbound_numba(f_fminbound, lower, upper),
        fminbound(f_fminbound, lower, upper, full_output=True),
    )


@numba.jit(nopython=True)
def f_fminbound(x):
    return x ** 2.0
