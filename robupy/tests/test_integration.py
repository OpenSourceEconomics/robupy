import numpy as np

from robupy.get_worst_case import get_worst_case_probs
from robupy.tests.get_random_testcase import get_request


def test_1():
    """This test just confirms that the package is running smoothly."""
    x, v, q, beta, gamma, is_cost = get_request()

    get_worst_case_probs(v, q, beta, is_cost)

