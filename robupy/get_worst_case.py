"""This module contains auxiliary functions for the robust optimization problem."""
import numba
import numpy as np

from robupy.checks import checks_get_worst_case_out
from robupy.checks import checks_get_worst_in
from robupy.config import EPS_FLOAT
from robupy.config import MAX_FLOAT
from robupy.minimize_scalar import fminbound_numba


@numba.jit(nopython=True)
def criterion_full(lambda_, v, v_max, q, beta):
    """This is the criterion function for solving the inner problem of Nilim and El
    Ghaoui (2003). It corresponds to equation (47) in the paper."""

    v_scaled = (v - v_max) / lambda_
    # We want to rule out an infinite logarithm.
    arg_ = np.maximum(np.sum(q * np.exp(v_scaled)), EPS_FLOAT)

    rslt = lambda_ * (np.log(arg_) + v_max / lambda_) + lambda_ * beta

    return rslt


@numba.jit(nopython=True)
def calculate_p(v, q, lambda_):
    """This function yields a closed form solution for the worst case distribution,
    given the solution to the inner problem lambda."""

    v_intern = v / lambda_ - np.max(v / lambda_)
    p = q * np.minimum(np.exp(v_intern), MAX_FLOAT)
    p = p / np.sum(p)

    return p


@numba.jit(nopython=True)
def get_worst_case_probs(v, q, beta, is_cost=True):
    """This function returns the worst distribution."""
    checks_get_worst_in(v, q, beta)

    # We want to handle two cases explicitly. First we deal with the case that there
    # is no ambiguity in the transition probabilities. Second, we look at the
    # case where the all mass assigned to the worst-case realization is inside the
    # feasible set.

    if beta == 0:
        return q.copy()
    elif beta >= -np.log(np.min(q)):
        p = np.zeros_like(q)
        if is_cost:
            p[np.argmax(v)] = 1
            return p
        else:
            p[np.argmin(v)] = 1
            return p

    # We can use this function to determine the worst case if we pass in costs or if
    # we pass in utility.
    if not is_cost:
        v_intern = -v
    else:
        v_intern = v

    v_max = np.max(v_intern)

    upper = np.maximum((v_max - np.dot(q, v_intern)) / beta, 2 * EPS_FLOAT)
    lower = EPS_FLOAT

    x, func_val, status, func_eval = fminbound_numba(
        criterion_full, lower, upper, args=(v_intern, v_max, q, beta), xatol=EPS_FLOAT
    )
    p = calculate_p(v_intern, q, x)

    checks_get_worst_case_out(p, q, beta, status)

    return p
