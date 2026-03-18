import torch
import torch.nn as nn
import torch.nn.functional as F

from spiking.spiking_module import SpikingModule
from spiking.threshold import ThresholdInitialization


class ConvIntegrateAndFireLayer(SpikingModule):
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
        super().__init__(
            num_inputs=in_channels * kernel_size * kernel_size,
            num_outputs=num_filters,
        )
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.refractory_period = refractory_period

        self.weights = nn.Parameter(
            torch.rand(
                (num_filters, in_channels, kernel_size, kernel_size), dtype=dtype
            )
        )
        self.thresholds = nn.Parameter(
            threshold_initialization.initialize((num_filters,))
        )

        # Spatial buffers are lazily initialized on first forward
        self._spatial_initialized = False
        self._dtype = dtype

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

    def _compute_output_size(self, H: int, W: int) -> tuple[int, int]:
        oH = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        oW = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return oH, oW

    def _update_refractory(self, dt: float) -> torch.Tensor:
        active = self.refractory_times == 0
        self.refractory_times.sub_(dt).clamp_(min=0.0)
        return active

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        C, H, W = incoming_spikes.shape
        oH, oW = self._compute_output_size(H, W)
        self._ensure_spatial_buffers(oH, oW)

        active = self._update_refractory(dt)
        self._output_spikes.zero_()

        if not active.any():
            return self._output_spikes

        # Check if any input spikes
        if not incoming_spikes.any():
            return self._output_spikes

        # Convolve input with filters
        contrib = F.conv2d(
            incoming_spikes.unsqueeze(0),
            self.weights,
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
            if self.training:
                self._output_spikes[crossed] = 1.0
            else:
                self._output_spikes[crossed] = 1.0
            self.membrane_potentials[crossed] = 0.0
            self._spike_times[crossed] = current_time
            self.refractory_times[crossed] = self.refractory_period

        return self._output_spikes

    def reset(self):
        if not self._spatial_initialized:
            return
        self.membrane_potentials.zero_()
        self.refractory_times.zero_()
        self._spike_times.fill_(float("inf"))
        self._output_spikes.zero_()

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times

    @torch.no_grad()
    def infer_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        """Compute first spike times analytically without mutating model state.

        Args:
            input_times: (C, H, W) tensor of spike times (inf = no spike).

        Returns:
            (F, oH, oW) tensor of output spike times.
        """
        C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)

        result = torch.full(
            (self.num_filters, oH, oW), float("inf"), dtype=input_times.dtype
        )

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return result

        unique_times = input_times[finite_mask].unique().sort()[0]

        cum_potential = torch.zeros((self.num_filters, oH, oW), dtype=input_times.dtype)
        not_yet_spiked = torch.ones((self.num_filters, oH, oW), dtype=torch.bool)

        for t in unique_times:
            active = (input_times == t).float()  # (C, H, W)
            contrib = F.conv2d(
                active.unsqueeze(0),
                self.weights,
                stride=self.stride,
                padding=self.padding,
            ).squeeze(
                0
            )  # (F, oH, oW)
            cum_potential += contrib

            crossed = (cum_potential >= self.thresholds.view(-1, 1, 1)) & not_yet_spiked
            result[crossed] = t
            not_yet_spiked &= ~crossed

            if not not_yet_spiked.any():
                break

        return result

    @torch.no_grad()
    def infer_spike_times_batch(self, input_times: torch.Tensor) -> torch.Tensor:
        """Batched analytical spike time inference.

        Args:
            input_times: (B, C, H, W) tensor of spike times.

        Returns:
            (B, F, oH, oW) tensor of output spike times.
        """
        B, C, H, W = input_times.shape
        oH, oW = self._compute_output_size(H, W)

        result = torch.full(
            (B, self.num_filters, oH, oW), float("inf"), dtype=input_times.dtype
        )

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return result

        unique_times = input_times[finite_mask].unique().sort()[0]

        cum_potential = torch.zeros(
            (B, self.num_filters, oH, oW), dtype=input_times.dtype
        )
        not_yet_spiked = torch.ones((B, self.num_filters, oH, oW), dtype=torch.bool)

        for t in unique_times:
            active = (input_times == t).float()  # (B, C, H, W)
            contrib = F.conv2d(
                active,
                self.weights,
                stride=self.stride,
                padding=self.padding,
            )  # (B, F, oH, oW)
            cum_potential += contrib

            crossed = (
                cum_potential >= self.thresholds.view(1, -1, 1, 1)
            ) & not_yet_spiked
            result[crossed] = t
            not_yet_spiked &= ~crossed

            if not not_yet_spiked.any():
                break

        return result
