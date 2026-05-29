from abc import ABC, abstractmethod

import torch


class Decoder(ABC):
    """Convert spike times to feature values."""

    @abstractmethod
    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        """Convert spike times to features.

        :param spike_times: (batch, neurons) tensor of spike times. Non-spiking
            neurons have inf values.
        :returns: (batch, neurons) tensor of feature values in [0, 1].
        """


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
