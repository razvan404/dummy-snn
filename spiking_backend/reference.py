from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _output_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1


def _affected_out_range(
    pos_padded: int, kernel: int, stride: int, out_size: int
) -> Tuple[int, int]:
    """Half-open output indices whose receptive field covers a padded input."""
    lo = max(0, -(-(pos_padded - kernel + 1) // stride))
    hi = min(out_size, pos_padded // stride + 1)
    return lo, hi


def spike_driven_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter reference; returns ``(spike_times, cum_potential)``."""
    B, C, H, W = input_times.shape
    F_, C_w, kH, kW = weights_4d.shape
    if C_w != C:
        raise ValueError(f"channel mismatch: weights C={C_w} vs input C={C}")
    if thresholds.numel() != F_:
        raise ValueError(f"thresholds size {thresholds.numel()} != F={F_}")

    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)
    spike_times = torch.full(
        (B, F_, oH, oW), float("inf"), dtype=input_times.dtype, device=input_times.device
    )
    cum_potential = torch.zeros(
        (B, F_, oH, oW), dtype=input_times.dtype, device=input_times.device
    )
    finite = torch.isfinite(input_times)
    if not finite.any():
        return spike_times, cum_potential

    unique_times = input_times[finite].unique().sort().values
    th_b = thresholds.view(1, -1, 1, 1)

    for t in unique_times:
        events = torch.nonzero(input_times == t, as_tuple=False)
        if events.numel() == 0:
            continue
        for s in range(events.shape[0]):
            b, c, y, x = (int(v) for v in events[s].tolist())
            yp, xp = y + padding, x + padding
            oh_lo, oh_hi = _affected_out_range(yp, kH, stride, oH)
            ow_lo, ow_hi = _affected_out_range(xp, kW, stride, oW)
            if oh_lo >= oh_hi or ow_lo >= ow_hi:
                continue
            for oh in range(oh_lo, oh_hi):
                ky = yp - oh * stride
                for ow in range(ow_lo, ow_hi):
                    kx = xp - ow * stride
                    cum_potential[b, :, oh, ow] += weights_4d[:, c, ky, kx]

        not_yet = torch.isinf(spike_times)
        crossed = not_yet & (cum_potential >= th_b)
        if crossed.any():
            spike_times = torch.where(
                crossed, torch.full_like(spike_times, float(t.item())), spike_times
            )
    return spike_times, cum_potential


def spike_driven_conv_accumulate_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Multi-threshold scatter reference; returns ``(K, B, F, oH, oW)``."""
    B, C, H, W = input_times.shape
    F_, C_w, kH, kW = weights_4d.shape
    K, F_th = thresholds_2d.shape
    if C_w != C or F_th != F_:
        raise ValueError("shape mismatch")

    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)
    spike_times = torch.full(
        (K, B, F_, oH, oW), float("inf"),
        dtype=input_times.dtype, device=input_times.device,
    )
    cum_potential = torch.zeros(
        (B, F_, oH, oW), dtype=input_times.dtype, device=input_times.device
    )
    finite = torch.isfinite(input_times)
    if not finite.any():
        return spike_times

    unique_times = input_times[finite].unique().sort().values
    th_b = thresholds_2d.view(K, 1, F_, 1, 1)

    for t in unique_times:
        events = torch.nonzero(input_times == t, as_tuple=False)
        if events.numel() == 0:
            continue
        for s in range(events.shape[0]):
            b, c, y, x = (int(v) for v in events[s].tolist())
            yp, xp = y + padding, x + padding
            oh_lo, oh_hi = _affected_out_range(yp, kH, stride, oH)
            ow_lo, ow_hi = _affected_out_range(xp, kW, stride, oW)
            if oh_lo >= oh_hi or ow_lo >= ow_hi:
                continue
            for oh in range(oh_lo, oh_hi):
                ky = yp - oh * stride
                for ow in range(ow_lo, ow_hi):
                    kx = xp - ow * stride
                    cum_potential[b, :, oh, ow] += weights_4d[:, c, ky, kx]

        not_yet = torch.isinf(spike_times)
        crossed = not_yet & (cum_potential.unsqueeze(0) >= th_b)
        if crossed.any():
            spike_times = torch.where(
                crossed, torch.full_like(spike_times, float(t.item())), spike_times
            )
    return spike_times


def _build_bins(
    input_times: torch.Tensor,
    weights_flat: torch.Tensor,
    num_bins: int,
    kH: int,
    kW: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Per-output bin histogram of weighted contributions: ``(B, F, L, num_bins)``."""
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
    """Gather-with-bins reference; ``cum_potential`` is empty unless requested."""
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
    """Multi-threshold gather reference; cum_potential shared across K."""
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
