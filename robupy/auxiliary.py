"""This module contains auxiliary functions for the robust optimization problem."""
from functools import partial

from scipy.optimize import fminbound
import numpy as np

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


def criterion_full(v, q, beta, lambda_):
    """This is the criterion function for ..."""
    checks_criterion_full_in(v, q, beta, lambda_)

    # We want to rule out an infinite logarithm.
    arg_ = np.clip(np.sum(q * np.exp(v / lambda_)), EPS_FLOAT, None)

    rslt = lambda_ * np.log(arg_) + lambda_ * beta

    checks_criterion_full_out(rslt)

    return rslt


def calculate_p(v, q, lambda_):
    """This function return the optimal ..."""
    checks_calculate_p_in(v, q, lambda_)

    p = q * np.clip(np.exp(v / lambda_), None, MAX_FLOAT)
    p = p / np.sum(p)

    checks_calculate_p_out(p)

    return p


def get_worst_case_probs(v, q, beta):
    """This function return the worst case measure."""
    checks_get_worst_in(v, q, beta)

    if beta == 0.0:
        return q.copy()

    # We scale the value function to avoid too large evaluations of the exponential function in
    # the calculate_p() function.
    v_scaled = v / max(abs(v))

    upper = np.clip(
        (max(v_scaled) - np.matmul(q, v_scaled)) / beta, 2 * EPS_FLOAT, None
    )
    lower = EPS_FLOAT

    criterion = partial(criterion_full, v_scaled, q, beta)

    x, func_val, status, func_eval = fminbound(
        criterion, lower, upper, xtol=EPS_FLOAT, full_output=True
    )
    p = calculate_p(v_scaled, q, x)

    checks_get_worst_case_out(p, q, beta, status)

    return p


def get_worst_case_outcome(v, q, beta, is_cost=True):
    """This function calculates the worst case outcome."""
    checks_get_worst_in(v, q, beta)

    # We can use this function to determine the worst case if we pass in costs or if we pass in
    # utility.
    if not is_cost:
        v_intern = -np.array(v).copy()
    else:
        v_intern = np.array(v)

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

    p = get_worst_case_probs(v_intern, q, beta)
    v_new = np.matmul(p, v)

    checks_get_worst_case_outcome_out(v, v_new)

    return v_new


def get_exponential_utility(x, gamma):
    """This function calculates the exponential utility."""
    return 1.0 / gamma * np.exp(-gamma * x)


def get_entropic_risk_measure(v, q, gamma):
    """This function calculates the entropic risk measure."""
    if gamma == 0:
        return -np.matmul(q, v)

    rslt = np.sum(q * np.exp(-gamma * v))
    rslt = (1.0 / gamma) * np.log(rslt)
    return rslt


def get_multiplier_evaluation(v, q, theta):
    """This function returns the evaluation based on the multiplier preferences."""

    def criterion_soft(v, q, gamma, epsilon):
        crit_val = (
            -get_worst_case_outcome(v, q, epsilon / gamma, is_cost=False) - epsilon
        )
        return -crit_val

    lower, upper = 0.00, MAX_FLOAT

    criterion = partial(criterion_soft, v, q, theta)
    rslt = fminbound(
        criterion, lower, upper, xtol=EPS_FLOAT, maxfun=MAX_INT, full_output=True
    )
    np.testing.assert_equal(rslt[2] == 0, True)

    return rslt[1]
