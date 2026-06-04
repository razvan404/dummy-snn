import math
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


class TargetRelativeSigmoid(Decoder):
    r"""Soft (sigmoid) variant of :class:`TargetRelative`.

    :class:`TargetRelative` decodes the target-relative latency
    ``z = (1 - t) / (1 - t_target)`` with a hard clamp to ``[0, 1]``. That clamp
    saturates: every spike at or before ``t_target`` collapses to exactly 1 and every
    spike at or after ``t = 1`` (including no spike) to exactly 0, so the feature is
    piecewise-constant over wide latency ranges. This variant replaces the clamp with a
    smooth logistic, grading those saturated regions:

        ``r = sigmoid(alpha * (z - 0.5))``

    ``alpha`` is the scaling factor (sharpness): as ``alpha -> inf`` the sigmoid
    approaches the hard clamp; smaller ``alpha`` grades more gently. A spike exactly at
    ``t_target`` sits at ``z = 1``.

    With ``normalize=True`` (default) the output is rescaled by the affine map sending the
    no-spike value ``sigmoid(-alpha/2)`` to 0 and the spike-at-``t_target`` value
    ``sigmoid(alpha/2)`` to 1, then clamped to ``[0, 1]`` — so the endpoints match
    :class:`TargetRelative` exactly (spike at the target -> 1, no spike -> 0) while the
    interior stays graded. With ``normalize=False`` the raw sigmoid is returned, so the
    no-spike value is ``sigmoid(-alpha/2) > 0``.
    """

    def __init__(self, t_target: float, alpha: float = 4.0, normalize: bool = True):
        self.t_target = t_target
        self.alpha = alpha
        self.normalize = normalize

    def decode(self, spike_times: torch.Tensor) -> torch.Tensor:
        z = (1.0 - spike_times) / (1.0 - self.t_target)
        r = torch.sigmoid(self.alpha * (z - 0.5))
        if self.normalize:
            s0 = 1.0 / (1.0 + math.exp(self.alpha / 2.0))   # sigmoid(-alpha/2): no-spike
            s1 = 1.0 / (1.0 + math.exp(-self.alpha / 2.0))  # sigmoid(+alpha/2): spike at t_target
            r = (r - s0) / (s1 - s0)
        return torch.clamp(r, min=0.0, max=1.0)
