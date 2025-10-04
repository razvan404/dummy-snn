import torch
from .spike import Spike


def iterate_spikes(spikes: list[Spike], shape: tuple[int, int, int]):
    """
    Iterates through a list of spikes.
    :param spikes: list of Spike objects (must be sorted by time).
    :param shape: tuple defining the shape of the spike volume.
    :return: generator yielding (incoming_spikes, current_time, delta_time)
    """
    spike_idx = 0
    prev_time = 0.0

    while spike_idx < len(spikes):
        current_time = spikes[spike_idx].time
        incoming_spikes = torch.zeros(shape, dtype=torch.float32)

        while (
            spike_idx < len(spikes)
            and (spike := spikes[spike_idx]).time <= current_time
        ):
            incoming_spikes[spike.z, spike.y, spike.x] = 1.0  # TODO: local patches here
            spike_idx += 1

        yield incoming_spikes, current_time, current_time - prev_time
        prev_time = current_time
