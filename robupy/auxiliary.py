"""This module contains auxiliary functions for the robust optimization problem."""
from functools import partial

from robupy.minimize import fminbound_numba
import numpy as np
import numba

from robupy.config import SMALL_FLOAT
from robupy.config import HUGE_FLOAT
from robupy.config import EPS_FLOAT
from robupy.config import MAX_FLOAT
from robupy.config import MAX_INT
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
    checks_criterion_full_in(v, q, beta, lambda_)

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
def get_worst_case_probs(v, q, beta):
    """This function return the worst case measure."""
    checks_get_worst_in(v, q, beta)

    if beta == 0.0:
        return q.copy()

    # We scale the value function to avoid too large evaluations of the exponential function in
    # the calculate_p() function.
    v_scaled = v / np.max(np.abs(v))

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

    # We can use this function to determine the worst case if we pass in costs or if
    # we pass in utility.
    if not is_cost:
        v_intern = -v
    else:
        v_intern = v

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

    p = get_worst_case_probs(v_intern, q, beta)
    v_new = np.dot(p, v)

    checks_get_worst_case_outcome_out(v, v_new)

    return v_new


@numba.jit(nopython=True)
def get_exponential_utility(x, gamma):
    """This function calculates the exponential utility."""
    return 1.0 / gamma * np.exp(-gamma * x)


@numba.jit(nopython=True)
def get_entropic_risk_measure(v, q, gamma):
    """This function calculates the entropic risk measure."""
    if gamma == 0:
        return -np.dot(q, v)

    rslt = np.sum(q * np.exp(-gamma * v))
    rslt = (1.0 / gamma) * np.log(rslt)
    return rslt


@numba.jit(nopython=True)
def get_multiplier_evaluation(v, q, theta):
    """This function returns the evaluation based on the multiplier preferences."""

    lower, upper = 0.00, MAX_FLOAT

    xf, fval, flag, num = fminbound_numba(
        criterion_soft,
        lower,
        upper,
        args=(v, q, theta),
        xatol=EPS_FLOAT,
        maxfun=MAX_INT,
    )

    if flag != 0:
        raise AssertionError
    return xf


@numba.jit(nopython=True)
def criterion_soft(epsilon, v, q, gamma):
    crit_val = -get_worst_case_outcome(v, q, epsilon / gamma, is_cost=False) - epsilon
    return -crit_val
