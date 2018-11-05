import numpy as np

from robupy.auxiliary import get_entropic_risk_measure
from robupy.auxiliary import get_exponential_utility
from robupy.tests.auxiliary import get_request


def test_1():
    """This test establishes that the calculation of the certainty equivalent is properly done."""
    for _ in range(50):

        _, v, q, beta, gamma, _ = get_request()

        # We scale the utilities as for very large values there are numerical discrepancies
        # driven by numerical errors.
        v = v * 0.1

        eu = 0.0
        for i, prob in enumerate(q):
            eu += prob * get_exponential_utility(v[i], gamma)

        rslt = get_exponential_utility(-get_entropic_risk_measure(v, q, gamma), gamma)

        np.testing.assert_almost_equal(rslt, eu)
