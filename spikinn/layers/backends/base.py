from __future__ import annotations

import torch
import torch.nn.functional as F


def base(
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
    membrane_out = torch.zeros(
        (B, num_filters, oH, oW), dtype=dtype, device=dev,
    )
    th = thresholds.view(-1, 1, 1)

    for b in range(B):
        sample = input_times[b]
        finite = torch.isfinite(sample)
        if not finite.any():
            continue
        unique_times = sample[finite].unique().sort()[0]
        membrane = torch.zeros((num_filters, oH, oW), dtype=dtype, device=dev)
        spiked = torch.zeros((num_filters, oH, oW), dtype=torch.bool, device=dev)
        for t in unique_times:
            frame = (sample == t).to(dtype).unsqueeze(0)
            contrib = F.conv2d(
                frame, weights_4d, stride=stride, padding=padding,
            ).squeeze(0)
            not_spiked = ~spiked
            membrane[not_spiked] += contrib[not_spiked]
            crossed = (membrane >= th) & not_spiked
            if crossed.any():
                spike_times[b][crossed] = t
                membrane[crossed] = 0.0
                spiked |= crossed
        membrane_out[b] = membrane
    return spike_times, membrane_out
