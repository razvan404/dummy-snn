import torch

from .spike import Spike


def convert_to_spikes(times: torch.Tensor) -> list[Spike]:
    finite_mask = torch.isfinite(times)
    if not finite_mask.any():
        return []

    indices = torch.nonzero(finite_mask, as_tuple=False)  # (N, 3) for k, i, j
    spike_times = times[finite_mask]

    sort_order = spike_times.argsort()
    indices = indices[sort_order]
    spike_times = spike_times[sort_order]

    return [
        Spike(x=idx[2].item(), y=idx[1].item(), z=idx[0].item(), time=t.item())
        for idx, t in zip(indices, spike_times)
    ]
