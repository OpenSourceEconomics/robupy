"""This module contains auxiliary functions for the robust optimization problem."""

from robupy.minimize import fminbound_numba
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

    # We want to rule out an infinite logarithm.
    arg_ = np.maximum(np.sum(q * np.exp(v / lambda_)), EPS_FLOAT)

    rslt = lambda_ * np.log(arg_) + lambda_ * beta

    checks_criterion_full_out(rslt)

    return rslt


@numba.jit(nopython=True)
def calculate_p(v, q, lambda_):
    """This function return the optimal ..."""
    checks_calculate_p_in(v, q, lambda_)

    p = q * np.minimum(np.exp(v / lambda_), MAX_FLOAT)
    p = p / np.sum(p)

    checks_calculate_p_out(p)

    return p


@numba.jit(nopython=True)
def get_worst_case_probs(v, q, beta, is_cost=True):
    """This function return the worst case measure."""
    checks_get_worst_in(v, q, beta)

    if (beta == 0.0) | (len(q) == 1):
        return q.copy()

    # We can use this function to determine the worst case if we pass in costs or if
    # we pass in utility.
    if not is_cost:
        v_intern = -v
    else:
        v_intern = v

    # We scale the value function to avoid too large evaluations of the exponential
    # function in
    # the calculate_p() function.
    v_scaled = v_intern / np.max(np.abs(v_intern))

    upper = np.maximum((np.max(v_scaled) - np.dot(q, v_scaled)) / beta, 2 * EPS_FLOAT)
    lower = EPS_FLOAT

    x, func_val, status, func_eval = fminbound_numba(
        criterion_full, lower, upper, args=(v_scaled, q, beta), xatol=EPS_FLOAT
    )
    p = calculate_p(v_scaled, q, x)

    checks_get_worst_case_out(p, q, beta, status)

    return p


@numba.jit(nopython=True)
def get_worst_case_outcome(v, q, beta, is_cost=True):
    """This function calculates the worst case outcome."""
    checks_get_worst_in(v, q, beta)

    # We want to handle two cases explicitly. First we deal with the case that there
    # is no ambiguity in the transition probabilities. Second, we look at the
    # case where the all mass assigned to the worst-case realization is inside the
    # feasible set.
    if beta == 0:
        return np.dot(q, v)
    elif beta >= np.max(-np.log(q)):
        if is_cost:
            return np.max(v)
        else:
            return np.min(v)

    p = get_worst_case_probs(v, q, beta, is_cost=is_cost)
    v_new = np.dot(p, v)

    checks_get_worst_case_outcome_out(v, v_new)

    return v_new
