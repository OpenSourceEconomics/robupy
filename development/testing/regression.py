#!/usr/bin/env python
"""This module is a first take at regression tests."""
import pickle as pkl
import numpy as np
import socket
import sys

from robupy.auxiliary import get_multiplier_evaluation
from robupy.tests.auxiliary import get_request

num_tests = 1000

if False:

    tests = []
    for _ in range(num_tests):
        print(_)
        x, v, q, beta, gamma, is_cost = get_request()
        args = v, q, gamma
        rslt = get_multiplier_evaluation(*args)
        tests.append([rslt, args])

    pkl.dump(tests, open('regression_vault.robupy.pkl', 'wb'))


fname = '../../robupy/tests/regression_vault.robupy.pkl'
#fname = 'regression_vault.robupy.pkl'

tests = pkl.load(open(fname, 'rb'))
for test in tests:
    rslt, args = test

    # We need to be less strict when the tests were created on a different machine.
    if 'heracles' in socket.gethostname():
        decimal = 7
    else:
        decimal = 5

    np.testing.assert_almost_equal(get_multiplier_evaluation(*args), rslt, decimal=decimal)
