from abc import ABC, abstractmethod

import torch


class Decoder(ABC):
    """Convert spike times to feature values."""

    @abstractmethod
    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        """Convert spike times to features.

        Args:
            spike_times: (batch, neurons) tensor of spike times. Non-spiking
                neurons have inf values.

        Returns:
            (batch, neurons) tensor of feature values in [0, 1].
        """


class BinaryFirstSpike(Decoder):
    """Neurons with the earliest finite spike time get 1.0, rest get 0.0."""

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        min_times = spike_times.min(dim=-1, keepdim=True).values
        result = torch.where(
            torch.isfinite(spike_times) & (spike_times == min_times),
            torch.ones_like(spike_times),
            torch.zeros_like(spike_times),
        )
        return result


class BinaryWindowFirstSpike(Decoder):
    """Neurons within ±tolerance of the earliest spike time get 1.0, rest get 0.0."""

    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        filled = spike_times.clone()
        filled[~torch.isfinite(filled)] = float("inf")
        min_times = filled.min(dim=-1, keepdim=True).values
        return torch.where(
            torch.isfinite(spike_times) & (spike_times <= min_times + self.tolerance),
            torch.ones_like(spike_times),
            torch.zeros_like(spike_times),
        )


class LinearInversion(Decoder):
    """clamp(1 - t, 0, 1). Earlier spikes yield higher features."""

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - spike_times, min=0, max=1.0)


class ScaledInversion(Decoder):
    """Per-sample scaling: clamp((1 - t) / (1 - min_t), 0, 1).

    The earliest spike per sample always maps to 1.0.
    """

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(spike_times)
        filled = spike_times.clone()
        filled[~finite_mask] = float("inf")
        min_t = filled.min(dim=-1, keepdim=True).values

        denom = 1.0 - min_t
        # Avoid division by zero when min_t == 1.0; avoid -inf denom when all inf
        safe_denom = torch.where(
            torch.isfinite(denom) & (denom != 0),
            denom,
            torch.ones_like(denom),
        )

        raw = (1.0 - spike_times) / safe_denom
        # Non-finite inputs map to 0
        raw = torch.where(finite_mask, raw, torch.zeros_like(raw))
        return torch.clamp(raw, min=0, max=1.0)


class TargetRelative(Decoder):
    """Falez Eq 10: clamp(1 - (t - t_target) / (1 - t_target), 0, 1)."""

    def __init__(self, t_target: float):
        self.t_target = t_target

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            1.0 - (spike_times - self.t_target) / (1.0 - self.t_target),
            min=0,
            max=1.0,
        )


class NeuronMeanRelative(Decoder):
    """Decode relative to per-neuron mean spike times: clamp(1 - (t - mean), 0, 1)."""

    def __init__(self, mean_spike_times: torch.Tensor):
        self.mean_spike_times = mean_spike_times

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        raw = 1.0 - (spike_times - self.mean_spike_times)
        return torch.clamp(raw, min=0, max=1.0)
