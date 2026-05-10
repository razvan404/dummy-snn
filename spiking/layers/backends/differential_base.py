from __future__ import annotations

import torch
import torch.nn.functional as F


def differential_base(
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
    """STE per-sample loop; same gradient semantics as ``differential_dense``."""
    B, _, H, W = input_times.shape
    num_filters, _, kH, kW = weights_4d.shape
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kW) // stride + 1
    dev, dtype = weights_4d.device, weights_4d.dtype

    hard_spike = torch.full(
        (B, num_filters, oH, oW), float("inf"), dtype=dtype, device=dev,
    )
    soft_spike = torch.zeros(
        (B, num_filters, oH, oW), dtype=dtype, device=dev,
    )
    cum_potential = torch.zeros(
        (B, num_filters, oH, oW), dtype=dtype, device=dev,
    )
    theta = thresholds.view(1, -1, 1, 1)
    weights = weights_4d.detach()

    for b in range(B):
        sample = input_times[b : b + 1]
        finite = torch.isfinite(sample)
        if not finite.any():
            continue
        unique_times = sample[finite].unique().sort()[0]
        T = len(unique_times)
        cum = torch.zeros((1, num_filters, oH, oW), dtype=dtype, device=dev)
        not_yet_spiked = torch.ones(
            (1, num_filters, oH, oW), dtype=torch.bool, device=dev,
        )
        soft_b = torch.zeros((1, num_filters, oH, oW), dtype=dtype, device=dev)

        for k in range(T):
            with torch.no_grad():
                active = (sample == unique_times[k]).to(dtype)
                cum = cum + F.conv2d(
                    active, weights, stride=stride, padding=padding,
                )
                crossed = (cum >= theta) & not_yet_spiked
                hard_spike[b : b + 1][crossed] = unique_times[k]
                not_yet_spiked = not_yet_spiked & ~crossed

            p_k = torch.sigmoid((cum - theta) / tau)
            next_t = unique_times[k + 1].item() if k < T - 1 else t_no_spike
            soft_b = soft_b + p_k * (unique_times[k].item() - next_t)

        soft_spike[b : b + 1] = soft_b
        cum_potential[b : b + 1] = cum.detach()

    soft_spike = soft_spike + t_no_spike
    spike_times = hard_spike + (soft_spike - soft_spike.detach())
    return spike_times, cum_potential
