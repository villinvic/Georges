import numpy as np


def log_uniform(low=0, high=1, size=None, base=10):
    assert low < high
    low = np.log(low + 1e-8)/np.log(base)
    high = np.log(high + 1e-8)/np.log(base)
    return np.power(base, np.random.uniform(low, high, size))