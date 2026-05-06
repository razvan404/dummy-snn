"""Spike-driven inference kernels.

Drop-in replacements for the dense ``F.conv2d``-per-unique-time loop used
by ``spiking.layers.ConvIntegrateAndFireLayer._conv2d_accumulate`` and
``applications.threshold_research.conv_neuron_perturbation.multi_threshold_conv_accumulate``.

The replacement exploits the fact that latency-encoded inputs are highly
sparse: ~1-2% of input slots fire per timestep, and once an output position
spikes no further updates affect it.

Backend selection is automatic: the compiled C++ / CUDA extension is used
when the input device is supported, otherwise we fall back to the pure
PyTorch reference (``reference.py``) which is correct but slow.

Public API:
    spike_driven_conv_accumulate(input_times, weights_4d, thresholds,
                                 stride, padding) -> (spike_times, cum_potential)
    spike_driven_conv_accumulate_multi_threshold(...)
    is_compiled_available() -> bool
"""

from __future__ import annotations

from typing import Tuple

import torch

from . import reference

try:
    from .build import load_backend

    _ext = load_backend()
except Exception:  # pragma: no cover
    _ext = None


def is_compiled_available() -> bool:
    """True iff the compiled extension finished loading."""
    return _ext is not None


def _has_cpu_fn() -> bool:
    return _ext is not None and hasattr(_ext, "spike_driven_conv_accumulate_cpu")


def _has_cuda_fn() -> bool:
    return _ext is not None and hasattr(_ext, "spike_driven_conv_accumulate_cuda")


def spike_driven_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse-event replacement for ``_conv2d_accumulate``.

    Returns ``(spike_times, cum_potential)`` both ``(B, F, oH, oW)``.
    """
    dev = input_times.device
    if dev.type == "cuda" and _has_cuda_fn():
        return _ext.spike_driven_conv_accumulate_cuda(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds.contiguous(),
            int(stride),
            int(padding),
        )
    if dev.type == "cpu" and _has_cpu_fn():
        return _ext.spike_driven_conv_accumulate_cpu(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds.contiguous(),
            int(stride),
            int(padding),
        )
    return reference.spike_driven_conv_accumulate(
        input_times, weights_4d, thresholds, stride=stride, padding=padding
    )


def spike_driven_conv_accumulate_multi_threshold(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Sparse-event replacement for ``multi_threshold_conv_accumulate``.

    Returns ``(K, B, F, oH, oW)`` spike times.
    """
    dev = input_times.device
    if dev.type == "cuda" and _ext is not None and hasattr(
        _ext, "spike_driven_conv_accumulate_multi_threshold_cuda"
    ):
        return _ext.spike_driven_conv_accumulate_multi_threshold_cuda(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds_2d.contiguous(),
            int(stride),
            int(padding),
        )
    if dev.type == "cpu" and _ext is not None and hasattr(
        _ext, "spike_driven_conv_accumulate_multi_threshold_cpu"
    ):
        return _ext.spike_driven_conv_accumulate_multi_threshold_cpu(
            input_times.contiguous(),
            weights_4d.contiguous(),
            thresholds_2d.contiguous(),
            int(stride),
            int(padding),
        )
    return reference.spike_driven_conv_accumulate_multi_threshold(
        input_times, weights_4d, thresholds_2d, stride=stride, padding=padding
    )


__all__ = [
    "spike_driven_conv_accumulate",
    "spike_driven_conv_accumulate_multi_threshold",
    "is_compiled_available",
]
