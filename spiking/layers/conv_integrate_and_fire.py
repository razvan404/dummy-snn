import torch
import torch.nn as nn
import torch.nn.functional as F

from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
from spiking.threshold import ThresholdInitialization


class ConvIntegrateAndFireLayer(IntegrateAndFireLayer):
    """Conv IF layer with pluggable analytical inference backends.

    Spike times in, spike times out. Stateless by default; caches
    ``_spike_times`` only when ``self.training`` (STDP requires B=1).
    """

    num_bins: int = 64
    tau: float = 1.0
    t_no_spike: float = 1.0

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        threshold_initialization: ThresholdInitialization = None,
        refractory_period: float = 1.0,
        dtype: torch.dtype = torch.float32,
        backend: str = "gather",
    ):
        from spiking.layers.backends import BACKENDS

        if backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}"
            )
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__(
            num_inputs=in_channels * kernel_size * kernel_size,
            num_outputs=num_filters,
            threshold_initialization=threshold_initialization,
            refractory_period=refractory_period,
            dtype=dtype,
        )
        self._oH: int | None = None
        self._oW: int | None = None
        self._dtype = dtype
        self._backend = backend
        self._spike_times: torch.Tensor | None = None

    @property
    def weights_4d(self) -> torch.Tensor:
        return self.weights.view(
            self.num_filters, self.in_channels, self.kernel_size, self.kernel_size
        )

    def _compute_output_size(self, H: int, W: int) -> tuple[int, int]:
        oH = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        oW = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return oH, oW

    def _unfold_patches(self, input_times: torch.Tensor) -> torch.Tensor:
        """Unfold to ``(L, dim)`` or ``(B, L, dim)``; pads with +inf so absent inputs don't look like t=0 spikes."""
        has_batch = input_times.dim() == 4
        if not has_batch:
            input_times = input_times.unsqueeze(0)
        if self.padding > 0:
            input_times = F.pad(input_times, [self.padding] * 4, value=float("inf"))
        patches = F.unfold(
            input_times,
            kernel_size=self.kernel_size,
            padding=0,
            stride=self.stride,
        )
        patches = patches.permute(0, 2, 1)
        return patches if has_batch else patches.squeeze(0)

    def forward(
        self,
        input_times: torch.Tensor,
        first_spike_only: bool = True,
    ) -> torch.Tensor:
        """Spike times in, spike times out: ``(B, C, H, W) -> (B, F, oH, oW)``.

        Potentials are not computed here — STDP and threshold adaptation read
        only spike times. Use ``infer_spike_times_and_potentials_batch`` when
        you explicitly need cumulative potentials.
        """
        from spiking.layers.backends import is_differentiable

        if input_times.dim() != 4:
            raise ValueError(
                f"forward expects (B, C, H, W); got shape {tuple(input_times.shape)}"
            )
        diff = is_differentiable(self._backend)
        if self.training and not diff and input_times.shape[0] != 1:
            raise ValueError(
                f"training requires B=1 for STDP backends, got B={input_times.shape[0]}"
            )

        ctx = torch.enable_grad if diff else torch.no_grad
        with ctx():
            spike_times, _ = self._dispatch_backend(
                input_times,
                with_cum_potential=False,
            )
            if first_spike_only:
                spike_times = self._wta_across_filters(spike_times)

        if self.training and not diff:
            self._spike_times = spike_times[0].detach()

        return spike_times

    def _dispatch_backend(
        self,
        input_times: torch.Tensor,
        *,
        with_cum_potential: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from spiking.layers.backends import get_backend

        return get_backend(self._backend)(
            input_times,
            self.weights_4d,
            self.thresholds,
            stride=self.stride,
            padding=self.padding,
            num_bins=self.num_bins,
            with_cum_potential=with_cum_potential,
            tau=self.tau,
            t_no_spike=self.t_no_spike,
        )

    @staticmethod
    def _wta_across_filters(spike_times: torch.Tensor) -> torch.Tensor:
        """Earliest filter wins per ``(b, oh, ow)``; ties broken uniformly at random."""
        min_time = spike_times.amin(dim=1, keepdim=True)
        candidates = (spike_times == min_time) & torch.isfinite(spike_times)
        rand = torch.rand_like(spike_times)
        rand = torch.where(candidates, rand, torch.full_like(rand, float("-inf")))
        winner_idx = rand.argmax(dim=1, keepdim=True)
        winner_mask = torch.zeros_like(spike_times, dtype=torch.bool)
        winner_mask.scatter_(1, winner_idx, True)
        winner_mask = winner_mask & candidates.any(dim=1, keepdim=True)
        return torch.where(
            winner_mask, spike_times, torch.full_like(spike_times, float("inf"))
        )

    @torch.no_grad()
    def _dense_conv2d_accumulate(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from spiking.layers.backends.dense import dense

        return dense(
            input_times,
            self.weights_4d,
            self.thresholds,
            stride=self.stride,
            padding=self.padding,
        )

    def _init_spatial_buffers(self, oH: int, oW: int) -> None:
        self._oH = oH
        self._oW = oW
        dev = self.weights.device
        self.register_buffer(
            "membrane_potentials",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype, device=dev),
        )
        self.register_buffer(
            "refractory_times",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype, device=dev),
        )
        self.register_buffer(
            "_step_spike_times",
            torch.full(
                (self.num_filters, oH, oW), float("inf"), dtype=self._dtype, device=dev
            ),
        )
        self.register_buffer(
            "_output_spikes",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype, device=dev),
        )

    def simulate_step(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        """One time-frame of step-by-step simulation."""
        if self._oH is None:
            H, W = incoming_spikes.shape[-2], incoming_spikes.shape[-1]
            oH, oW = self._compute_output_size(H, W)
            self._init_spatial_buffers(oH, oW)

        active = self.refractory_times == 0
        self.refractory_times.sub_(dt).clamp_(min=0.0)
        self._output_spikes.zero_()
        if not active.any():
            return self._output_spikes
        if not incoming_spikes.any():
            return self._output_spikes

        contrib = F.conv2d(
            incoming_spikes.unsqueeze(0),
            self.weights_4d,
            stride=self.stride,
            padding=self.padding,
        ).squeeze(0)

        update_mask = active & torch.isinf(self._step_spike_times)
        self.membrane_potentials[update_mask] += contrib[update_mask]
        crossed = (
            self.membrane_potentials >= self.thresholds.view(-1, 1, 1)
        ) & update_mask
        if crossed.any():
            self._output_spikes[crossed] = 1.0
            self.membrane_potentials[crossed] = 0.0
            self._step_spike_times[crossed] = current_time
            self.refractory_times[crossed] = self.refractory_period
        return self._output_spikes

    def reset(self):
        self._spike_times = None
        if self._oH is not None:
            self.membrane_potentials.zero_()
            self.refractory_times.zero_()
            self._step_spike_times.fill_(float("inf"))
            self._output_spikes.zero_()
            self._oH = None
            self._oW = None

    @property
    def spike_times(self) -> torch.Tensor | None:
        if self._spike_times is not None:
            return self._spike_times
        return getattr(self, "_step_spike_times", None)

    @torch.no_grad()
    def _conv2d_accumulate(
        self,
        input_times: torch.Tensor,
        *,
        with_cum_potential: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dispatch_backend(
            input_times,
            with_cum_potential=with_cum_potential,
        )

    @torch.no_grad()
    def infer_spike_times_batch(self, input_times: torch.Tensor) -> torch.Tensor:
        if input_times.dim() == 2:
            return super().infer_spike_times_batch(input_times)
        spike_times, _ = self._conv2d_accumulate(input_times, with_cum_potential=False)
        return spike_times

    @torch.no_grad()
    def infer_spike_times_and_potentials_batch(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_times.dim() == 2:
            return super().infer_spike_times_and_potentials_batch(input_times)
        return self._conv2d_accumulate(input_times, with_cum_potential=True)

    @torch.no_grad()
    def infer_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        """Unbatched analytical inference via unfold + base FC."""
        if input_times.dim() == 1:
            return super().infer_spike_times(input_times)
        C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)
        patches = self._unfold_patches(input_times)
        result = super().infer_spike_times_batch(patches)
        return result.view(oH, oW, self.num_filters).permute(2, 0, 1)

    @torch.no_grad()
    def infer_spike_times_batch_unfold(self, input_times: torch.Tensor) -> torch.Tensor:
        B, C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)
        L = oH * oW
        patches = self._unfold_patches(input_times)
        flat = patches.reshape(B * L, -1)
        flat_result = super().infer_spike_times_batch(flat)
        return flat_result.view(B, oH, oW, self.num_filters).permute(0, 3, 1, 2)
