import torch

from .base import BaseUnsupervisedTrainer
from spikinn.layers.sequential import SpikinnSequential


class ConvUnsupervisedTrainer(BaseUnsupervisedTrainer):
    def _prepare_input(self, times: torch.Tensor) -> torch.Tensor:
        return times.to(self.device)

    def _forward_analytical(self, prepared: torch.Tensor) -> None:
        if isinstance(self.model, SpikinnSequential):
            super()._forward_analytical(prepared)
            return
        batched = prepared.unsqueeze(0)
        self.model(batched, first_spike_only=False)
