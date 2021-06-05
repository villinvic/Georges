import numpy as np


def log_uniform(low=0, high=1, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))