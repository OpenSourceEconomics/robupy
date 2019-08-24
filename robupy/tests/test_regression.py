"""This module contains the regression tests."""
# import pickle as pkl
#
# import numpy as np
#
# from robupy.auxiliary import get_multiplier_evaluation
# from robupy.config import PACKAGE_DIR


# def test_1():
#     """This tests runs a couple of regression tests."""
#     tests = pkl.load(open(PACKAGE_DIR +
#     '/tests/regression_vault.robupy.pkl', 'rb'))[:5]
#     for test in tests:
#         rslt, args = test
#         np.testing.assert_almost_equal(get_multiplier_evaluation(*args), rslt)

from numpy.testing import assert_array_almost_equal
from robupy.auxiliary import (
    get_worst_case_outcome,
    get_entropic_risk_measure,
    get_exponential_utility,
    get_worst_case_probs,
)
from robupy.tests.pre_numba.auxiliary import (
    pre_numba_get_entropic_risk_measure,
    pre_numba_get_worst_case_outcome,
    pre_numba_get_exponential_utility,
    pre_numba_get_worst_case_probs,
)
from robupy.tests.auxiliary import get_request


def test_2():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        pre_numba_get_worst_case_outcome(v, q, beta, is_cost),
        get_worst_case_outcome(v, q, beta, is_cost),
    )


def test_3():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        get_entropic_risk_measure(v, q, gamma),
        pre_numba_get_entropic_risk_measure(v, q, gamma),
    )


def test_4():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        get_exponential_utility(x, gamma), pre_numba_get_exponential_utility(x, gamma)
    )


def test_5():
    x, v, q, beta, gamma, is_cost = get_request()
    assert_array_almost_equal(
        get_worst_case_probs(v, q, beta), pre_numba_get_worst_case_probs(v, q, beta)
    )


# def test_5():
#     x, v, q, beta, gamma, is_cost = get_request()
# assert_array_almost_equal(
#     pre_numba_get_multiplier_evaluation(v, q, gamma),
#     get_multiplier_evaluation(v, q, gamma),
# )
