import torch

from .spike import Spike


def convert_to_spikes(times: torch.Tensor) -> list[Spike]:
    spikes = []
    for k in range(times.shape[0]):
        for i in range(times.shape[1]):
            for j in range(times.shape[2]):
                time = times[k, i, j]
                if not torch.isinf(time):
                    spikes.append(Spike(x=j, y=i, z=k, time=time.item()))

    return sorted(spikes, key=lambda spike: spike.time)
