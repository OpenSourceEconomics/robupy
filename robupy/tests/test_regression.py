"""This module contains the regression tests."""
import pickle as pkl

import numpy as np

from robupy.auxiliary import get_multiplier_evaluation
from robupy.config import PACKAGE_DIR


# def test_1():
#     """This tests runs a couple of regression tests."""
#     tests = pkl.load(open(PACKAGE_DIR + '/tests/regression_vault.robupy.pkl', 'rb'))[:5]
#     for test in tests:
#         rslt, args = test
#         np.testing.assert_almost_equal(get_multiplier_evaluation(*args), rslt)