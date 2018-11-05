#!/usr/bin/env python
import random

import numpy as np

from robupy.tests.test_integration import test_1
from robupy.tests.test_integration import test_2
from robupy.tests.test_integration import test_3


while True:

    for i, test in enumerate([test_1, test_2, test_3]):

        seed = random.randrange(1, 100000)
        np.random.seed(seed)

        try:
            test()
        except:
            print('\n failure in test ' + str(i + 1) + ' for seed ', seed, '\n')
            raise AssertionError