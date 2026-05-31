from __future__ import annotations

import torch
import torch.nn.functional as F


def dense(
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
    B, _, H, W = input_times.shape
    num_filters, _, kH, kW = weights_4d.shape
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kW) // stride + 1
    dev, dtype = weights_4d.device, weights_4d.dtype

    spike_times = torch.full(
        (B, num_filters, oH, oW), float("inf"), dtype=dtype, device=dev,
    )
    cum_potential = torch.zeros(
        (B, num_filters, oH, oW), dtype=dtype, device=dev,
    )
    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return spike_times, cum_potential

    unique_times = input_times[finite_mask].unique().sort()[0]
    not_yet_spiked = torch.ones(
        (B, num_filters, oH, oW), dtype=torch.bool, device=dev,
    )
    for t in unique_times:
        active = (input_times == t).float()
        contrib = F.conv2d(active, weights_4d, stride=stride, padding=padding)
        cum_potential += contrib
        crossed = (
            cum_potential >= thresholds.view(1, -1, 1, 1)
        ) & not_yet_spiked
        spike_times[crossed] = t
        not_yet_spiked &= ~crossed
        if not not_yet_spiked.any():
            break
    return spike_times, cum_potential
