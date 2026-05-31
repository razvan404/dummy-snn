from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _output_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1


def _build_bins(
    input_times: torch.Tensor,
    weights_flat: torch.Tensor,
    num_bins: int,
    kH: int,
    kW: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    B, _, _, _ = input_times.shape
    F_, rf = weights_flat.shape
    if padding > 0:
        input_times = F.pad(input_times, [padding] * 4, value=float("inf"))
    patches = F.unfold(input_times, kernel_size=(kH, kW), stride=stride)
    L = patches.size(-1)

    finite = torch.isfinite(patches)
    bin_idx = torch.where(
        finite,
        (patches * num_bins).long().clamp(0, num_bins - 1),
        torch.full_like(patches, num_bins, dtype=torch.long),
    )

    bins = torch.zeros(
        (B, F_, L, num_bins + 1), dtype=patches.dtype, device=patches.device
    )
    for r in range(rf):
        idx = bin_idx[:, r, :].unsqueeze(1).expand(-1, F_, -1).unsqueeze(-1)
        w = weights_flat[:, r].view(1, F_, 1, 1).expand(B, F_, L, 1)
        bins.scatter_add_(3, idx, w)
    return bins[..., :num_bins]


def first_spike_times_gather(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    num_bins: int,
    stride: int = 1,
    padding: int = 0,
    compute_cum_potential: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = input_times.shape
    F_, C_w, kH, kW = weights_4d.shape
    if C_w != C:
        raise ValueError("channel mismatch")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)

    weights_flat = weights_4d.reshape(F_, C * kH * kW)
    real_bins = _build_bins(input_times, weights_flat, num_bins, kH, kW, stride, padding)
    cum = real_bins.cumsum(dim=-1)
    th = thresholds.view(1, F_, 1, 1)
    crossed = cum >= th
    any_cross = crossed.any(dim=-1)
    first = crossed.float().argmax(dim=-1)
    spike_t = torch.where(
        any_cross,
        first.to(input_times.dtype) / num_bins,
        torch.full_like(first.to(input_times.dtype), float("inf")),
    ).view(B, F_, oH, oW)
    if compute_cum_potential:
        pot = real_bins.sum(dim=-1).view(B, F_, oH, oW)
    else:
        pot = torch.empty(0, dtype=input_times.dtype, device=input_times.device)
    return spike_t, pot


def first_spike_times_gather_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    num_bins: int,
    stride: int = 1,
    padding: int = 0,
    compute_cum_potential: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = input_times.shape
    F_, _, kH, kW = weights_4d.shape
    K, F_th = thresholds_2d.shape
    if F_th != F_:
        raise ValueError(f"threshold last dim {F_th} != F={F_}")
    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)

    weights_flat = weights_4d.reshape(F_, C * kH * kW)
    real_bins = _build_bins(input_times, weights_flat, num_bins, kH, kW, stride, padding)
    cum = real_bins.cumsum(dim=-1)
    th = thresholds_2d.view(K, 1, F_, 1, 1)
    crossed = cum.unsqueeze(0) >= th
    any_cross = crossed.any(dim=-1)
    first = crossed.float().argmax(dim=-1)
    spike_t = torch.where(
        any_cross,
        first.to(input_times.dtype) / num_bins,
        torch.full_like(first.to(input_times.dtype), float("inf")),
    ).view(K, B, F_, oH, oW)
    if compute_cum_potential:
        pot = real_bins.sum(dim=-1).view(B, F_, oH, oW)
    else:
        pot = torch.empty(0, dtype=input_times.dtype, device=input_times.device)
    return spike_t, pot
