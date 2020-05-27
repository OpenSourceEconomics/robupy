"""This module contains all checks for the package to improve readability."""
import numba
import numpy as np


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
def check_p(p):
    if np.abs(np.sum(p) - 1) > 1e-08:
        raise AssertionError
    if not np.all(p >= 0.0):
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
