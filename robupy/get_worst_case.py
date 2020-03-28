"""This module contains auxiliary functions for the robust optimization problem."""

from robupy.minimize_scalar import fminbound_numba
import numpy as np
import numba

from robupy.config import EPS_FLOAT
from robupy.config import MAX_FLOAT
from robupy.checks import (
    checks_criterion_full_in,
    checks_criterion_full_out,
    checks_calculate_p_in,
    checks_calculate_p_out,
    checks_get_worst_in,
    checks_get_worst_case_outcome_out,
    checks_get_worst_case_out,
)


@numba.jit(nopython=True)
def criterion_full(lambda_, v, q, beta):
    """This is the criterion function for ..."""
    checks_criterion_full_in(lambda_)

    v_max = np.max(v / lambda_)
    v_scaled = v / lambda_ - v_max
    # We want to rule out an infinite logarithm.
    arg_ = np.maximum(np.sum(q * np.exp(v_scaled)), EPS_FLOAT)

    rslt = lambda_ * (np.log(arg_) + v_max) + lambda_ * beta

    checks_criterion_full_out(rslt)

    return rslt


@numba.jit(nopython=True)
def calculate_p(v, q, lambda_):
    """This function return the optimal ..."""
    checks_calculate_p_in(v, q, lambda_)

    v_intern = v / lambda_ - np.max(v / lambda_)
    p = q * np.minimum(np.exp(v_intern), MAX_FLOAT)
    p = p / np.sum(p)

    checks_calculate_p_out(p)

    return p


@numba.jit(nopython=True)
def get_worst_case_probs(v, q, beta, is_cost=True):
    """This function return the worst case measure."""
    checks_get_worst_in(v, q, beta)

    # We want to handle two cases explicitly. First we deal with the case that there
    # is no ambiguity in the transition probabilities. Second, we look at the
    # case where the all mass assigned to the worst-case realization is inside the
    # feasible set.

    if beta == 0 or len(q) == 1:
        return q.copy()
    elif beta >= np.max(-np.log(q)):
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

    upper = np.maximum((np.max(v_intern) - np.dot(q, v_intern)) / beta, 2 * EPS_FLOAT)
    lower = EPS_FLOAT

    x, func_val, status, func_eval = fminbound_numba(
        criterion_full, lower, upper, args=(v_intern, q, beta), xatol=EPS_FLOAT
    )
    p = calculate_p(v_intern, q, x)

    checks_get_worst_case_out(p, q, beta, status)

    return p


@numba.jit(nopython=True)
def get_worst_case_outcome(v, q, beta, is_cost=True):
    """This function calculates the worst case outcome."""

    p = get_worst_case_probs(v, q, beta, is_cost=is_cost)
    v_new = np.dot(p, v)

    checks_get_worst_case_outcome_out(v, v_new)

    return v_new
