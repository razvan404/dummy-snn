from __future__ import annotations

from typing import Tuple

import torch

from . import reference

try:
    from . import _ext as _ext_module
    _ext = _ext_module
except ImportError:
    try:
        from .build import load_backend
        _ext = load_backend()
    except Exception:
        _ext = None


def is_compiled_available() -> bool:
    return _ext is not None


def _has(name: str) -> bool:
    return _ext is not None and hasattr(_ext, name)


def _dispatch(input_times: torch.Tensor, cuda_fn: str, cpu_fn: str):
    if input_times.device.type == "cuda" and _has(cuda_fn):
        return getattr(_ext, cuda_fn)
    if input_times.device.type == "cpu" and _has(cpu_fn):
        return getattr(_ext, cpu_fn)
    return None


def spike_driven_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    num_bins: int = 64,
    compute_cum_potential: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse-event scatter; ``compute_cum_potential=False`` lets it skip already-spiked outputs."""
    is_cuda = input_times.device.type == "cuda"
    fn = _dispatch(
        input_times,
        "spike_driven_conv_accumulate_cuda",
        "spike_driven_conv_accumulate_cpu",
    )
    if fn is None:
        return reference.spike_driven_conv_accumulate(
            input_times, weights_4d, thresholds, stride=stride, padding=padding
        )
    args = [
        input_times.contiguous(),
        weights_4d.contiguous(),
        thresholds.contiguous(),
        int(stride),
        int(padding),
    ]
    if not is_cuda:
        args.append(int(num_bins))
    args.append(bool(compute_cum_potential))
    return fn(*args)


def spike_driven_conv_accumulate_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    num_bins: int = 64,
) -> torch.Tensor:
    """Scatter multi-threshold; returns ``(K, B, F, oH, oW)``."""
    is_cuda = input_times.device.type == "cuda"
    fn = _dispatch(
        input_times,
        "spike_driven_conv_accumulate_multi_threshold_cuda",
        "spike_driven_conv_accumulate_multi_threshold_cpu",
    )
    if fn is None:
        return reference.spike_driven_conv_accumulate_multi_threshold(
            input_times, weights_4d, thresholds_2d, stride=stride, padding=padding
        )
    args = [
        input_times.contiguous(),
        weights_4d.contiguous(),
        thresholds_2d.contiguous(),
        int(stride),
        int(padding),
    ]
    if not is_cuda:
        args.append(int(num_bins))
    return fn(*args)


def _wta_per_position(spike_times: torch.Tensor) -> torch.Tensor:
    """Earliest filter wins per ``(b, oh, ow)``."""
    winner_f = spike_times.argmin(dim=1, keepdim=True)
    F_ = spike_times.size(1)
    f_idx = torch.arange(F_, device=spike_times.device).view(1, F_, 1, 1)
    keep = f_idx == winner_f
    return torch.where(keep, spike_times, torch.full_like(spike_times, float("inf")))


def first_spike_times(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    num_bins: int = 64,
    stride: int = 1,
    padding: int = 0,
    wta: bool = False,
    compute_cum_potential: bool = False,
):
    """Gather first-spike: per-output bin-histogram + prefix-sum; quantises to ``1/num_bins``."""
    fn = _dispatch(
        input_times, "first_spike_times_cuda", "first_spike_times_cpu"
    )
    if fn is None:
        out, pot = reference.first_spike_times_gather(
            input_times, weights_4d, thresholds,
            num_bins=num_bins, stride=stride, padding=padding,
            compute_cum_potential=compute_cum_potential,
        )
    else:
        out, pot = fn(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds.contiguous(),
            int(num_bins),
            int(stride),
            int(padding),
            bool(compute_cum_potential),
        )
    if wta:
        out = _wta_per_position(out)
    return (out, pot) if compute_cum_potential else out


def first_spike_times_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    num_bins: int = 64,
    stride: int = 1,
    padding: int = 0,
    compute_cum_potential: bool = False,
):
    """Gather first-spike multi-threshold; cum_potential is shared across K."""
    fn = _dispatch(
        input_times,
        "first_spike_times_multi_threshold_cuda",
        "first_spike_times_multi_threshold_cpu",
    )
    if fn is None:
        out, pot = reference.first_spike_times_gather_multi_threshold(
            input_times, weights_4d, thresholds_2d,
            num_bins=num_bins, stride=stride, padding=padding,
            compute_cum_potential=compute_cum_potential,
        )
    else:
        out, pot = fn(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds_2d.contiguous(),
            int(num_bins),
            int(stride),
            int(padding),
            bool(compute_cum_potential),
        )
    return (out, pot) if compute_cum_potential else out


__all__ = [
    "spike_driven_conv_accumulate",
    "spike_driven_conv_accumulate_multi_threshold",
    "first_spike_times",
    "first_spike_times_multi_threshold",
    "is_compiled_available",
]
