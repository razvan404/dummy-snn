import torch
import torch.nn as nn
import torch.nn.functional as F

from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
from spiking.threshold import ThresholdInitialization


class ConvIntegrateAndFireLayer(IntegrateAndFireLayer):
    """Convolutional integrate-and-fire layer.

    Inherits from IntegrateAndFireLayer and reuses its inference logic by
    unfolding spatial inputs into patches. Weights are stored as 2D
    (num_filters, C*kH*kW) and viewed as 4D for conv2d in forward().
    """

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
    ):
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

        # Spatial buffers are lazily initialized on first forward
        self._spatial_initialized = False
        self._dtype = dtype

    @property
    def weights_4d(self) -> torch.Tensor:
        """View weights as (num_filters, C, kH, kW) for conv2d operations."""
        return self.weights.view(
            self.num_filters, self.in_channels, self.kernel_size, self.kernel_size
        )

    def _compute_output_size(self, H: int, W: int) -> tuple[int, int]:
        oH = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        oW = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return oH, oW

    def _ensure_spatial_buffers(self, oH: int, oW: int):
        """Create spatial buffers on first forward pass when output size is known."""
        if self._spatial_initialized:
            return
        self.register_buffer(
            "membrane_potentials",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype),
        )
        self.register_buffer(
            "refractory_times",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype),
        )
        self.register_buffer(
            "_spike_times",
            torch.full((self.num_filters, oH, oW), float("inf"), dtype=self._dtype),
        )
        self.register_buffer(
            "_output_spikes",
            torch.zeros((self.num_filters, oH, oW), dtype=self._dtype),
        )
        self._oH = oH
        self._oW = oW
        self._spatial_initialized = True

    def _unfold_patches(self, input_times: torch.Tensor) -> torch.Tensor:
        """Extract patches from spatial input.

        :param input_times: (C, H, W) or (B, C, H, W) tensor of spike times.
        :returns: (L, dim) or (B, L, dim) tensor of unfolded patches,
            where L = oH * oW and dim = C * kH * kW.
        """
        has_batch = input_times.dim() == 4
        if not has_batch:
            input_times = input_times.unsqueeze(0)

        # Pad with inf (no-spike) before unfolding, since F.unfold pads with 0
        if self.padding > 0:
            input_times = F.pad(input_times, [self.padding] * 4, value=float("inf"))
            pad_for_unfold = 0
        else:
            pad_for_unfold = 0

        # F.unfold: (B, C, H, W) -> (B, C*kH*kW, L)
        patches = F.unfold(
            input_times,
            kernel_size=self.kernel_size,
            padding=pad_for_unfold,
            stride=self.stride,
        )  # (B, dim, L)
        patches = patches.permute(0, 2, 1)  # (B, L, dim)

        if not has_batch:
            patches = patches.squeeze(0)  # (L, dim)
        return patches

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        C, H, W = incoming_spikes.shape
        oH, oW = self._compute_output_size(H, W)
        self._ensure_spatial_buffers(oH, oW)

        active = self.refractory_times == 0
        self.refractory_times.sub_(dt).clamp_(min=0.0)
        self._output_spikes.zero_()

        if not active.any():
            return self._output_spikes

        if not incoming_spikes.any():
            return self._output_spikes

        # Convolve input with filters using 4D weight view
        contrib = F.conv2d(
            incoming_spikes.unsqueeze(0),
            self.weights_4d,
            stride=self.stride,
            padding=self.padding,
        ).squeeze(
            0
        )  # (F, oH, oW)

        # Accumulate only for active, not-yet-spiked neurons
        update_mask = active & torch.isinf(self._spike_times)
        self.membrane_potentials[update_mask] += contrib[update_mask]

        # Check threshold crossing
        crossed = (
            self.membrane_potentials >= self.thresholds.view(-1, 1, 1)
        ) & update_mask
        if crossed.any():
            self._output_spikes[crossed] = 1.0
            self.membrane_potentials[crossed] = 0.0
            self._spike_times[crossed] = current_time
            self.refractory_times[crossed] = self.refractory_period

        return self._output_spikes

    def reset(self):
        if not self._spatial_initialized:
            super().reset()
            return
        self.membrane_potentials.zero_()
        self.refractory_times.zero_()
        self._spike_times.fill_(float("inf"))
        self._output_spikes.zero_()

    def reset_spatial(self):
        """Reset spatial buffers so they are re-created for a new input size."""
        self._spatial_initialized = False

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times

    @torch.no_grad()
    def infer_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        """Compute first spike times analytically.

        Handles both spatial (C, H, W) and flat (num_inputs,) inputs.
        Flat inputs delegate to the base class FC inference (useful for
        patch-based training where patches are flattened).

        :param input_times: (C, H, W) or (num_inputs,) tensor of spike times.
        :returns: (F, oH, oW) for spatial input, (num_outputs,) for flat input.
        """
        if input_times.dim() == 1:
            return super().infer_spike_times(input_times)
        C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)

        patches = self._unfold_patches(input_times)  # (L, dim)
        # Delegate to base class batch inference: (L, dim) -> (L, F)
        result = super().infer_spike_times_batch(patches)  # (L, F)
        # Reshape to spatial: (L, F) -> (oH, oW, F) -> (F, oH, oW)
        return result.view(oH, oW, self.num_filters).permute(2, 0, 1)

    @torch.no_grad()
    def _conv2d_accumulate(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core conv2d accumulation loop shared by batch inference methods.

        :param input_times: (B, C, H, W) tensor of spike times.
        :returns: (spike_times, cum_potential) both (B, F, oH, oW).
        """
        B, C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)
        dev = input_times.device

        result = torch.full(
            (B, self.num_filters, oH, oW), float("inf"),
            dtype=input_times.dtype, device=dev,
        )
        cum_potential = torch.zeros(
            (B, self.num_filters, oH, oW), dtype=input_times.dtype, device=dev,
        )

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return result, cum_potential

        unique_times = input_times[finite_mask].unique().sort()[0]

        not_yet_spiked = torch.ones(
            (B, self.num_filters, oH, oW), dtype=torch.bool, device=dev,
        )

        for t in unique_times:
            active = (input_times == t).float()
            contrib = F.conv2d(
                active,
                self.weights_4d,
                stride=self.stride,
                padding=self.padding,
            )
            cum_potential += contrib

            crossed = (
                cum_potential >= self.thresholds.view(1, -1, 1, 1)
            ) & not_yet_spiked
            result[crossed] = t
            not_yet_spiked &= ~crossed

            if not not_yet_spiked.any():
                break

        return result, cum_potential

    @torch.no_grad()
    def infer_spike_times_and_potentials_batch(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analytical spike time inference using conv2d, also returning potentials.

        :param input_times: (B, C, H, W) tensor of spike times.
        :returns: (spike_times, cum_potential) both (B, F, oH, oW).
        """
        if input_times.dim() == 2:
            # Called from base class delegation (e.g. infer_spike_times on patches)
            return super().infer_spike_times_and_potentials_batch(input_times)
        return self._conv2d_accumulate(input_times)

    @torch.no_grad()
    def infer_spike_times_batch(self, input_times: torch.Tensor) -> torch.Tensor:
        """Batched analytical spike time inference using conv2d.

        :param input_times: (B, C, H, W) tensor of spike times.
        :returns: (B, F, oH, oW) tensor of output spike times.
        """
        if input_times.dim() == 2:
            # Called from base class delegation (e.g. infer_spike_times on patches)
            return super().infer_spike_times_batch(input_times)
        result, _ = self._conv2d_accumulate(input_times)
        return result

    @torch.no_grad()
    def infer_spike_times_batch_unfold(self, input_times: torch.Tensor) -> torch.Tensor:
        """Batched inference via patch unfolding (delegates to base class matmul).

        Slower than conv2d-based infer_spike_times_batch but useful as a
        reference implementation that directly reuses the base class logic.

        :param input_times: (B, C, H, W) tensor of spike times.
        :returns: (B, F, oH, oW) tensor of output spike times.
        """
        B, C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)
        L = oH * oW

        patches = self._unfold_patches(input_times)  # (B, L, dim)
        flat_patches = patches.reshape(B * L, -1)
        flat_result = super().infer_spike_times_batch(flat_patches)  # (B*L, F)
        return flat_result.view(B, oH, oW, self.num_filters).permute(0, 3, 1, 2)
