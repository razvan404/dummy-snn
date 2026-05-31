from abc import ABC, abstractmethod

import torch


class Decoder(ABC):
    @abstractmethod
    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        ...


class ScaledInversion(Decoder):
    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(spike_times)
        filled = spike_times.clone()
        filled[~finite_mask] = float("inf")
        min_t = filled.min(dim=-1, keepdim=True).values

        denom = 1.0 - min_t
        safe_denom = torch.where(
            torch.isfinite(denom) & (denom != 0),
            denom,
            torch.ones_like(denom),
        )

        raw = (1.0 - spike_times) / safe_denom
        raw = torch.where(finite_mask, raw, torch.zeros_like(raw))
        return torch.clamp(raw, min=0, max=1.0)


class TargetRelative(Decoder):
    def __init__(self, t_target: float):
        self.t_target = t_target

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            1.0 - (spike_times - self.t_target) / (1.0 - self.t_target),
            min=0,
            max=1.0,
        )
