"""This module contains all checks for the package to improve readability."""
import numpy as np
import numba

from robupy.config import SMALL_FLOAT


@numba.jit(nopython=True)
def checks_criterion_full_in(lambda_):
    """This function hosts all checks."""
    check_lambda(lambda_)


@numba.jit(nopython=True)
def checks_criterion_full_out(rslt):
    # We do encounter the case that the criterion function is infinite. However, this
    # is not a problem, since it is only called in a minimization which discards this
    # evaluation point.
    if not (np.isfinite(rslt) or rslt == np.inf):
        raise AssertionError


@numba.jit(nopython=True)
def checks_calculate_p_in(v, q, lambda_):
    check_lambda(lambda_)
    check_p(q)
    check_v(v)


@numba.jit(nopython=True)
def checks_calculate_p_out(p):
    check_p(p)


@numba.jit(nopython=True)
def checks_get_worst_in(v, q, beta):
    check_v(v)
    check_p(q)
    check_beta(beta)


@numba.jit(nopython=True)
def checks_get_worst_case_out(p, q, beta, status):
    check_p(q)
    check_p(p)
    check_beta(beta)
    if status != 0:
        raise AssertionError


@numba.jit(nopython=True)
def checks_get_worst_case_outcome_out(v, v_new):
    if not ((np.min(v) - SMALL_FLOAT <= v_new) & (v_new <= max(v) + SMALL_FLOAT)):
        raise AssertionError


@numba.jit(nopython=True)
def check_p(p):
    if np.abs(np.sum(p) - 1) > 1e-08:
        raise AssertionError
    if not np.all(p >= 0.0):
        raise AssertionError


@numba.jit(nopython=True)
def check_lambda(lambda_):
    if lambda_ < 0:
        raise AssertionError


@numba.jit(nopython=True)
def check_beta(beta):
    if beta < 0:
        raise AssertionError


@numba.jit(nopython=True)
def check_v(v):
    if not np.all(np.isfinite(v)):
        raise AssertionError


@numba.jit(nopython=True)
def check_kullback_leibler_entropy(p, q, beta, atol=0.0001):
    if len(p) != len(q):
        raise AssertionError
    if np.sum(np.log(np.multiply(p, np.divide(p, q)))) - beta > atol:
        raise AssertionError
