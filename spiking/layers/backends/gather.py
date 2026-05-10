from __future__ import annotations

import torch


def _empty_pot(spike_times: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, dtype=spike_times.dtype, device=spike_times.device)


def gather(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    stride: int = 1,
    padding: int = 0,
    num_bins: int = 64,
    with_cum_potential: bool = True,
    tau: float = 1.0,
    t_no_spike: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output bin-histogram scan; spike times at bin granularity (``k / num_bins``)."""
    from spiking_backend import first_spike_times

    out = first_spike_times(
        input_times, weights_4d, thresholds,
        num_bins=num_bins, stride=stride, padding=padding,
        compute_cum_potential=with_cum_potential,
    )
    if with_cum_potential:
        return out
    return out, _empty_pot(out)
