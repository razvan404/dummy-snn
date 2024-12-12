import numpy as np

from .spike import Spike


def convert_to_spikes(times: np.ndarray) -> list[Spike]:
    spikes = []
    for k in range(times.shape[0]):
        for i in range(times.shape[1]):
            for j in range(times.shape[2]):
                if (time := times[k][i][j]) != np.inf:
                    spikes.append(Spike(i, j, k, time))
    return sorted(spikes, key=lambda spike: spike.time)
