"""This module contains auxiliary functions for the robust optimization problem."""
from functools import partial

import numpy as np
from scipy.optimize import fminbound

from robupy.tests.resources.pre_numba.checks import pre_numba_checks
from robupy.tests.resources.pre_numba.config import EPS_FLOAT
from robupy.tests.resources.pre_numba.config import MAX_FLOAT


def pre_numba_criterion_full(v, q, beta, lambda_):
    """This is the criterion function for ..."""
    pre_numba_checks("criterion_full_in", v, q, beta, lambda_)

    # We want to rule out an infinite logarithm.
    arg_ = np.clip(np.sum(q * np.exp(v / lambda_)), EPS_FLOAT, None)

    rslt = lambda_ * np.log(arg_) + lambda_ * beta

    pre_numba_checks("criterion_full_out", rslt)

    return rslt


def pre_numba_calculate_p(v, q, lambda_):
    """This function return the optimal ..."""
    pre_numba_checks("calculate_p_in", v, q, lambda_)

    p = q * np.clip(np.exp(v / lambda_), None, MAX_FLOAT)
    p = p / np.sum(p)

    pre_numba_checks("calculate_p_out", p)

    return p


def pre_numba_get_worst_case_probs(v, q, beta, is_cost=True):
    """This function return the worst case measure."""
    pre_numba_checks("get_worst_case_in", v, q, beta)

    if beta == 0.0:
        return q.copy()

    # We can use this function to determine the worst case if we pass in costs or if we pass in
    # utility.
    if not is_cost:
        v_intern = -np.array(v).copy()
    else:
        v_intern = np.array(v)

    # We scale the value function to avoid too large evaluations of the exponential function in
    # the calculate_p() function.
    v_scaled = v_intern / max(abs(v_intern))

    upper = np.clip(
        (max(v_scaled) - np.matmul(q, v_scaled)) / beta, 2 * EPS_FLOAT, None
    )
    lower = EPS_FLOAT

    criterion = partial(pre_numba_criterion_full, v_scaled, q, beta)

    rslt = fminbound(criterion, lower, upper, xtol=EPS_FLOAT, full_output=True)
    p = pre_numba_calculate_p(v_scaled, q, rslt[0])

    pre_numba_checks("get_worst_case_out", p, q, beta, rslt)

    return p


def pre_numba_get_worst_case_outcome(v, q, beta, is_cost=True):
    """This function calculates the worst case outcome."""
    pre_numba_checks("get_worst_case_outcome_in", v, q, beta)

    # We want to handle two cases explicitly. First we deal with the case that there is no
    # ambiguity in the transition probabilities. Second, we look at the case where the all mass
    # assigned to the worst-case realization is inside the feasible set.
    if beta == 0:
        return np.matmul(q, v)
    elif beta >= max(-np.log(q)):
        if is_cost:
            return max(v)
        else:
            return min(v)

    p = pre_numba_get_worst_case_probs(v, q, beta, is_cost)
    rslt = np.matmul(p, v)

    pre_numba_checks("get_worst_case_outcome_out", v, q, beta, is_cost, rslt)

    return rslt
