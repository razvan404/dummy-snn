"""Pure-PyTorch sparse-event reference implementation.

This module is correctness scaffolding for the C/CUDA backends — slow
(Python event loop) but deterministic and easy to read. Tests pin the
algorithm against the existing dense path
(``ConvIntegrateAndFireLayer._conv2d_accumulate``) before any compiled
kernel takes over.

Algorithm (single-threshold):

  1. Identify finite spike events in the input tensor.
  2. Process events grouped by unique time, in ascending time order.
  3. For each event at time t and input position (b, c, y, x), scatter the
     contribution ``W[:, c, ky, kx]`` to every output position ``(b, :,
     oh, ow)`` whose ``kH×kW`` receptive field covers (y, x). The kernel
     indices ``ky = y + padding - oh*stride``, ``kx = x + padding -
     ow*stride``.
  4. Once *all* contributions for time t have been added to the cumulative
     potential, check threshold crossings — only positions that hadn't
     spiked yet record their spike time as t.

Step 4's "accumulate all then check" semantics are what
``_conv2d_accumulate`` does (one ``F.conv2d`` per unique time, then one
threshold check). Reproducing them lets us match output up to float
non-associativity.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _output_size(in_size: int, kernel: int, stride: int, padding: int) -> int:
    return (in_size + 2 * padding - kernel) // stride + 1


def _affected_out_range(
    pos_padded: int, kernel: int, stride: int, out_size: int
) -> Tuple[int, int]:
    """Output indices whose receptive field covers padded input position.

    Returns half-open ``[lo, hi)``.
    """
    lo = max(0, -(-(pos_padded - kernel + 1) // stride))  # ceil-div
    hi = min(out_size, pos_padded // stride + 1)
    return lo, hi


def spike_driven_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse-event reference for ``_conv2d_accumulate``.

    :param input_times: ``(B, C, H, W)`` spike times (``inf`` = no spike).
    :param weights_4d: ``(F, C, kH, kW)`` filter weights.
    :param thresholds: ``(F,)`` per-filter firing threshold.
    :param stride: conv stride.
    :param padding: conv padding (the padded slots are treated as ``inf``,
        i.e. they never spike).
    :returns: ``(spike_times, cum_potential)`` both ``(B, F, oH, oW)``.
    """
    B, C, H, W = input_times.shape
    F_, C_w, kH, kW = weights_4d.shape
    if C_w != C:
        raise ValueError(f"channel mismatch: weights C={C_w} vs input C={C}")
    if thresholds.numel() != F_:
        raise ValueError(
            f"thresholds size {thresholds.numel()} != num_filters {F_}"
        )

    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)
    dev = input_times.device
    dt = input_times.dtype

    spike_times = torch.full(
        (B, F_, oH, oW), float("inf"), dtype=dt, device=dev
    )
    cum_potential = torch.zeros((B, F_, oH, oW), dtype=dt, device=dev)

    finite = torch.isfinite(input_times)
    if not finite.any():
        return spike_times, cum_potential

    unique_times = input_times[finite].unique().sort().values  # ascending
    thresholds_b = thresholds.view(1, -1, 1, 1)

    for t in unique_times:
        # All events firing at exactly this time step.
        events = torch.nonzero(input_times == t, as_tuple=False)  # (S_t, 4)
        if events.numel() == 0:
            continue

        for s in range(events.shape[0]):
            b, c, y, x = (int(v) for v in events[s].tolist())
            y_p, x_p = y + padding, x + padding
            oh_lo, oh_hi = _affected_out_range(y_p, kH, stride, oH)
            ow_lo, ow_hi = _affected_out_range(x_p, kW, stride, oW)
            if oh_lo >= oh_hi or ow_lo >= ow_hi:
                continue

            for oh in range(oh_lo, oh_hi):
                ky = y_p - oh * stride
                for ow in range(ow_lo, ow_hi):
                    kx = x_p - ow * stride
                    # Add the per-filter contribution for this event.
                    cum_potential[b, :, oh, ow] += weights_4d[:, c, ky, kx]

        # After all events for this time have been added, check crossings.
        not_yet = torch.isinf(spike_times)
        crossed = not_yet & (cum_potential >= thresholds_b)
        if crossed.any():
            spike_times = torch.where(
                crossed, torch.full_like(spike_times, float(t.item())), spike_times
            )
            if not torch.isinf(spike_times).any():
                break

    return spike_times, cum_potential


def spike_driven_conv_accumulate_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Sparse-event reference for ``multi_threshold_conv_accumulate``.

    Shares the cumulative-potential accumulation across the K threshold
    sets and only branches at the per-set crossing check.

    :param input_times: ``(B, C, H, W)`` spike times.
    :param weights_4d: ``(F, C, kH, kW)`` filter weights.
    :param thresholds_2d: ``(K, F)`` per-filter thresholds for K variants.
    :returns: ``(K, B, F, oH, oW)`` spike times.
    """
    B, C, H, W = input_times.shape
    F_, C_w, kH, kW = weights_4d.shape
    K, F_th = thresholds_2d.shape
    if F_th != F_:
        raise ValueError(
            f"thresholds last dim {F_th} != num_filters {F_}"
        )

    oH = _output_size(H, kH, stride, padding)
    oW = _output_size(W, kW, stride, padding)
    dev = input_times.device
    dt = input_times.dtype

    spike_times = torch.full(
        (K, B, F_, oH, oW), float("inf"), dtype=dt, device=dev
    )
    cum_potential = torch.zeros((B, F_, oH, oW), dtype=dt, device=dev)

    finite = torch.isfinite(input_times)
    if not finite.any():
        return spike_times

    unique_times = input_times[finite].unique().sort().values
    thresholds_b = thresholds_2d.view(K, 1, F_, 1, 1)

    for t in unique_times:
        events = torch.nonzero(input_times == t, as_tuple=False)
        if events.numel() == 0:
            continue

        for s in range(events.shape[0]):
            b, c, y, x = (int(v) for v in events[s].tolist())
            y_p, x_p = y + padding, x + padding
            oh_lo, oh_hi = _affected_out_range(y_p, kH, stride, oH)
            ow_lo, ow_hi = _affected_out_range(x_p, kW, stride, oW)
            if oh_lo >= oh_hi or ow_lo >= ow_hi:
                continue
            for oh in range(oh_lo, oh_hi):
                ky = y_p - oh * stride
                for ow in range(ow_lo, ow_hi):
                    kx = x_p - ow * stride
                    cum_potential[b, :, oh, ow] += weights_4d[:, c, ky, kx]

        # Broadcast threshold check across K variants.
        not_yet = torch.isinf(spike_times)  # (K, B, F, oH, oW)
        crossed = not_yet & (cum_potential.unsqueeze(0) >= thresholds_b)
        if crossed.any():
            spike_times = torch.where(
                crossed, torch.full_like(spike_times, float(t.item())), spike_times
            )
            if not torch.isinf(spike_times).any():
                break

    return spike_times
