from __future__ import annotations

import logging
from typing import Tuple

import torch

from . import reference

logger = logging.getLogger(__name__)

try:
    from . import _ext as _ext_module
    _ext = _ext_module
except ImportError:
    try:
        from .build import load_backend
        _ext = load_backend()
    except Exception:
        _ext = None
        logger.warning(
            "spikinn compiled backend unavailable; using the pure-PyTorch reference kernel "
            "(slower, and may differ in tie-break / accumulation order)."
        )


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


def _wta_per_position(spike_times: torch.Tensor) -> torch.Tensor:
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
    "first_spike_times",
    "first_spike_times_multi_threshold",
    "is_compiled_available",
]
