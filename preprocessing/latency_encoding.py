import numpy as np


def apply_latency_encoding(input_data: np.ndarray):
    times = np.maximum(0.0, 1.0 - input_data)
    times[times == 1.0] = np.inf
    return times
