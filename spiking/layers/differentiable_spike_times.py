"""Differentiable spike time computation via soft first-crossing.

Makes the threshold-to-spike-time mapping differentiable using a sigmoid-based
soft approximation of the first threshold crossing. Weights are frozen; only
thresholds receive gradients.

The analytical spike time is t_spike = first t_k such that V(t_k) >= θ, where
V(t_k) is the cumulative membrane potential. This is a step function of θ.

Soft approximation:
    p_k = σ((V_k - θ) / τ)   — soft CDF of having crossed by time k
    q_k = p_k - p_{k-1}      — probability mass at time k
    t_spike = Σ q_k * t_k + (1 - p_T) * t_no_spike

Summation-by-parts form (used in implementation):
    t_spike = t_no_spike + Σ_k p_k * (t_k - t_{k+1}) + p_T * (t_T - t_no_spike)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableConvSpikeTime(nn.Module):
    """Differentiable convolutional spike time layer.

    Computes soft spike times from input spike times using frozen conv weights
    and learnable thresholds. Gradients flow through the sigmoid soft-crossing
    into the thresholds.

    Supports two modes:
    - **Soft mode** (use_ste=False): Returns soft spike times. Classifier trains
      on soft features. Subject to soft-hard feature gap.
    - **STE mode** (use_ste=True): Returns hard spike times in forward pass,
      routes gradients through the soft sigmoid in backward pass. Eliminates
      soft-hard gap — classifier always sees real features.
    """

    def __init__(
        self,
        weights_4d: torch.Tensor,
        thresholds: torch.Tensor,
        stride: int = 1,
        padding: int = 0,
        tau: float = 1.0,
        t_no_spike: float = 1.0,
        use_ste: bool = False,
    ):
        """
        :param weights_4d: (F, C, kH, kW) frozen conv filter weights.
        :param thresholds: (F,) initial threshold values (will become nn.Parameter).
        :param stride: Conv stride.
        :param padding: Conv padding.
        :param tau: Temperature for sigmoid soft-crossing.
        :param t_no_spike: Spike time assigned to non-spiking neurons.
        :param use_ste: If True, use straight-through estimator.
        """
        super().__init__()
        self.register_buffer("weights_4d", weights_4d.detach().clone())
        self.thresholds = nn.Parameter(thresholds.detach().clone())
        self.stride = stride
        self.padding = padding
        self.tau = tau
        self.t_no_spike = t_no_spike
        self.use_ste = use_ste

    @property
    def num_filters(self) -> int:
        return self.weights_4d.shape[0]

    def forward(self, input_times: torch.Tensor) -> torch.Tensor:
        """Compute differentiable spike times.

        In soft mode: returns soft spike times (differentiable w.r.t. thresholds).
        In STE mode: returns hard spike times in forward, soft gradients in backward.

        :param input_times: (B, C, H, W) spike times (inf = no spike).
        :returns: (B, F, oH, oW) spike times, differentiable w.r.t. thresholds.
        """
        B, C, H, W = input_times.shape
        kH = self.weights_4d.shape[2]
        oH = (H + 2 * self.padding - kH) // self.stride + 1
        oW = (W + 2 * self.padding - kH) // self.stride + 1
        device = input_times.device

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            fill = float("inf") if self.use_ste else self.t_no_spike
            return torch.full(
                (B, self.num_filters, oH, oW),
                fill,
                dtype=input_times.dtype,
                device=device,
            )

        unique_times = input_times[finite_mask].unique().sort()[0]
        T = len(unique_times)

        cum_potential = torch.zeros(
            B, self.num_filters, oH, oW, dtype=input_times.dtype, device=device
        )
        theta_view = self.thresholds.view(1, -1, 1, 1)
        soft_spike = torch.zeros(
            B, self.num_filters, oH, oW, dtype=input_times.dtype, device=device
        )

        # STE mode: also track hard first-crossings
        if self.use_ste:
            hard_spike = torch.full(
                (B, self.num_filters, oH, oW),
                float("inf"),
                dtype=input_times.dtype,
                device=device,
            )
            not_yet_spiked = torch.ones(
                B, self.num_filters, oH, oW, dtype=torch.bool, device=device
            )

        for k in range(T):
            with torch.no_grad():
                active = (input_times == unique_times[k]).float()
                contrib = F.conv2d(
                    active, self.weights_4d, stride=self.stride, padding=self.padding
                )
                cum_potential = cum_potential + contrib

            # Soft crossing — sigmoid carries gradient through thresholds
            p_k = torch.sigmoid((cum_potential - theta_view) / self.tau)

            if k < T - 1:
                delta_t = unique_times[k].item() - unique_times[k + 1].item()
            else:
                delta_t = unique_times[k].item() - self.t_no_spike
            soft_spike = soft_spike + p_k * delta_t

            # Hard crossing detection (no grad)
            if self.use_ste:
                with torch.no_grad():
                    crossed = (cum_potential >= theta_view) & not_yet_spiked
                    hard_spike[crossed] = unique_times[k]
                    not_yet_spiked = not_yet_spiked & ~crossed

        soft_spike = soft_spike + self.t_no_spike

        if self.use_ste:
            # STE: hard values forward, soft gradients backward
            # hard_spike has no grad; soft_spike - soft_spike.detach() is zero
            # in forward but carries d(soft)/d(theta) in backward
            return hard_spike + (soft_spike - soft_spike.detach())

        return soft_spike

    @torch.no_grad()
    def hard_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        """Compute hard (non-differentiable) spike times for comparison.

        Mirrors ConvIntegrateAndFireLayer._conv2d_accumulate logic.
        """
        B, C, H, W = input_times.shape
        kH = self.weights_4d.shape[2]
        oH = (H + 2 * self.padding - kH) // self.stride + 1
        oW = (W + 2 * self.padding - kH) // self.stride + 1
        device = input_times.device

        result = torch.full(
            (B, self.num_filters, oH, oW),
            float("inf"),
            dtype=input_times.dtype,
            device=device,
        )
        cum_potential = torch.zeros_like(result)

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return result

        unique_times = input_times[finite_mask].unique().sort()[0]
        not_yet_spiked = torch.ones_like(result, dtype=torch.bool)

        for t in unique_times:
            active = (input_times == t).float()
            contrib = F.conv2d(
                active, self.weights_4d, stride=self.stride, padding=self.padding
            )
            cum_potential += contrib

            crossed = (
                cum_potential >= self.thresholds.view(1, -1, 1, 1)
            ) & not_yet_spiked
            result[crossed] = t
            not_yet_spiked &= ~crossed

            if not not_yet_spiked.any():
                break

        return result
