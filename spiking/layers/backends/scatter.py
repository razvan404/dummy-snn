from __future__ import annotations

import torch


def scatter(
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
    from spiking_backend import spike_driven_conv_accumulate

    return spike_driven_conv_accumulate(
        input_times, weights_4d, thresholds,
        stride=stride, padding=padding, num_bins=num_bins,
        compute_cum_potential=with_cum_potential,
    )
