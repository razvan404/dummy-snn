from __future__ import annotations

import torch
import torch.nn.functional as F

from spiking.spiking_module import SpikingModule


class SpikeTimeMinPool(SpikingModule):
    """First-spike-wins pooling: each output is the earliest (min) spike time in its window.

    min over a window = −max(−x), so `F.max_pool2d` handles ∞ correctly — a window of all
    non-firing (∞) inputs stays ∞. Spike-times in → spike-times out; channels unchanged.
    Sits between conv-IF layers; not first/last in a stack, so its num_inputs/outputs
    (which `SpikingSequential` reads only from the endpoints) are unused.
    """

    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__(num_inputs=0, num_outputs=0)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self._spike_times: torch.Tensor | None = None

    def _pool(self, times: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-times, self.kernel_size, self.stride)

    def infer_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        self._spike_times = self._pool(input_times.unsqueeze(0)).squeeze(0)
        return self._spike_times

    def infer_spike_times_batch(self, input_times: torch.Tensor) -> torch.Tensor:
        self._spike_times = self._pool(input_times)
        return self._spike_times

    @property
    def spike_times(self):
        return self._spike_times

    def reset(self):
        self._spike_times = None

    def simulate_step(self, incoming_spikes, current_time, dt):
        raise NotImplementedError("SpikeTimeMinPool supports analytical inference only")
