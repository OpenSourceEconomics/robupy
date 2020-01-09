"""This module provides the fixtures for the PYTEST runs."""
import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1223)
