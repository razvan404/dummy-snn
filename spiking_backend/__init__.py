"""Spike-driven inference kernels.

This package provides drop-in replacements for the dense
``F.conv2d``-per-unique-time loop used by
``spiking.layers.ConvIntegrateAndFireLayer._conv2d_accumulate`` and
``applications.threshold_research.conv_neuron_perturbation.multi_threshold_conv_accumulate``.

The replacement exploits the fact that latency-encoded inputs are highly
sparse: only the spike events (typically 1-2% of input slots per timestep)
contribute to the cumulative potential, and once an output position spikes
no further updates affect it.

Public API:
    spike_driven_conv_accumulate(input_times, weights_4d, thresholds,
                                 stride, padding) -> (spike_times, cum_potential)

Backends are tried in order: CUDA C++ (if available and input is on CUDA),
CPU C++ (if available), pure-PyTorch reference (always available).
"""

from .reference import spike_driven_conv_accumulate as _reference_impl
from .reference import (
    spike_driven_conv_accumulate_multi_threshold as _reference_impl_multi,
)


def spike_driven_conv_accumulate(*args, **kwargs):
    """See ``reference.spike_driven_conv_accumulate``."""
    return _reference_impl(*args, **kwargs)


def spike_driven_conv_accumulate_multi_threshold(*args, **kwargs):
    """See ``reference.spike_driven_conv_accumulate_multi_threshold``."""
    return _reference_impl_multi(*args, **kwargs)


__all__ = [
    "spike_driven_conv_accumulate",
    "spike_driven_conv_accumulate_multi_threshold",
]
