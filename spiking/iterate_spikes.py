import numpy as np

from .spike import Spike


def iterate_spikes(spikes: list[Spike], shape: (int, int, int)):
    """
    Iterates through a list of spikes.
    :param spikes: list of spikes.
    :param shape: the shape of the spikes.
    :return: a generator that generates tuples containing (incoming_spikes, current_time, delta_time)
    """
    spike_idx = 0
    prev_time = 0.0

    while spike_idx < len(spikes):
        current_time = spikes[spike_idx].time
        incoming_spikes = np.zeros(shape, dtype=np.float32)

        while (
            spike_idx < len(spikes)
            and (spike := spikes[spike_idx]).time <= current_time
        ):
            incoming_spikes[spike.z, spike.x, spike.y] = 1.0
            spike_idx += 1

        yield incoming_spikes, current_time, current_time - prev_time
        prev_time = current_time
