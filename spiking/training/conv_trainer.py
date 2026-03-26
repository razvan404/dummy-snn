import torch

from .base import BaseUnsupervisedTrainer


class ConvUnsupervisedTrainer(BaseUnsupervisedTrainer):
    """Trainer for convolutional spiking layers.

    Preserves spatial dimensions of input spike times.
    """

    def _prepare_input(self, times: torch.Tensor) -> torch.Tensor:
        return times.to(self.device)
