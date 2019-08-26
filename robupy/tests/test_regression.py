"""This module contains the regression tests."""

from numpy.testing import assert_array_almost_equal
from robupy.auxiliary import (
    get_worst_case_outcome,
    get_worst_case_probs,
)
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
        get_worst_case_probs(v, q, beta), pre_numba_get_worst_case_probs(v, q, beta)
    )
