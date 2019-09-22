import numpy as np
import os

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

EPS_FLOAT = np.finfo(1.0).eps
MAX_FLOAT = np.finfo(1.0).max
SMALL_FLOAT = 1.0e-6
HUGE_FLOAT = 1e250

MAX_INT = np.iinfo(1).max

