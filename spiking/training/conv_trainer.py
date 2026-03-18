from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from spiking import iterate_spikes
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.learning.conv_learner import ConvLearner


class ConvUnsupervisedTrainer:
    def __init__(
        self,
        model: ConvIntegrateAndFireLayer,
        learner: ConvLearner,
        image_shape: tuple[int, int, int],
        on_batch_end: Callable[[int, float, str], None] | None = None,
        early_stopping: bool = True,
    ):
        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.learner = learner
        self.image_shape = image_shape
        self.on_batch_end = on_batch_end
        self.early_stopping = early_stopping

    def step_batch(
        self,
        batch_idx: int,
        times: torch.Tensor,
        /,
        split: str = "train",
    ):
        times = times.to(self.device)
        with torch.no_grad():
            for incoming_spikes, current_time, dt in iterate_spikes(times):
                output_spikes = self.model.forward(incoming_spikes, current_time, dt)
                if self.early_stopping and torch.any(output_spikes == 1.0):
                    break
        dw = 0.0
        if self.model.training:
            dw = self.learner.step(times)

        if self.on_batch_end:
            self.on_batch_end(batch_idx, dw, split)
        self.model.reset()
        return dw

    def step_loader(self, loader: DataLoader, /, split: str = "train"):
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        for batch_idx, (times, _label) in enumerate(loader):
            self.step_batch(batch_idx, times, split=split)

    def step_epoch(self):
        self.learner.learning_rate_step()
