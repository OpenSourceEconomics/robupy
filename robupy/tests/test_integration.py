import numpy as np

from robupy.get_worst_case import get_worst_case_outcome
from robupy.get_worst_case import get_worst_case_probs
from robupy.tests.get_random_testcase import get_request


def test_1():
    """This test just confirms that the package is running smoothly."""
    x, v, q, beta, gamma, is_cost = get_request()

    get_worst_case_outcome(v, q, beta, is_cost)
    get_worst_case_probs(v, q, beta, is_cost)


def test_3():
    """This test ensures that the worst-case is increasing with the size of the
    uncertainty set."""
    _, v, q, beta, _, is_cost = get_request()

    rslt = list()
    for beta in np.linspace(0.0, 10):
        rslt.append(get_worst_case_outcome(v, q, beta, is_cost))

    if is_cost:
        cond = np.all(np.diff(np.around(rslt, 5)) >= 0)
    else:
        cond = np.all(np.diff(np.around(rslt, 5)) <= 0)

    np.testing.assert_equal(cond, True)
