import numpy as np

from robupy.auxiliary import get_multiplier_evaluation
from robupy.auxiliary import get_entropic_risk_measure
from robupy.auxiliary import get_exponential_utility
from robupy.auxiliary import get_worst_case_outcome
from robupy.auxiliary import get_worst_case_probs
from robupy.tests.auxiliary import get_request


def test_1():
    """This test just confirms that the package is running smoothly."""
    x, v, q, beta, gamma, is_cost = get_request()

    get_worst_case_outcome(v, q, beta, is_cost)
    # get_multiplier_evaluation(v, q, gamma)
    get_entropic_risk_measure(v, q, gamma)
    get_exponential_utility(x, gamma)
    get_worst_case_probs(v, q, beta)


# def test_2():
#     """This test ensures the equivalence between the multiplier evaluation and the entropic risk
#     measure."""
#     _, v, q, _, gamma, _ = get_request()
#
#     theta = 1.0 / gamma
#
#     rslt_risk = get_entropic_risk_measure(v, q, gamma)
#     rslt_mult = get_multiplier_evaluation(v, q, theta)
#
#     np.testing.assert_almost_equal(-rslt_risk, rslt_mult, decimal=4)


def test_3():
    """This test ensures that the worst-case is increasing with the size of the uncertainty set."""
    _, v, q, beta, _, is_cost = get_request()

    rslt = list()
    for beta in np.linspace(0.0, 10):
        rslt.append(get_worst_case_outcome(v, q, beta, is_cost))

    if is_cost:
        cond = np.all(np.diff(np.around(rslt, 5)) >= 0)
    else:
        cond = np.all(np.diff(np.around(rslt, 5)) <= 0)

    np.testing.assert_equal(cond, True)
