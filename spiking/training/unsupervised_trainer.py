from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from spiking import iterate_spikes
from spiking.layers.sequential import SpikingSequential
from spiking.learning.learner import Learner
from spiking.spiking_module import SpikingModule


class UnsupervisedTrainer:
    def __init__(
        self,
        model: SpikingModule,
        learner: Learner,
        image_shape: (int, int, int),
        on_batch_end: Callable[[int, float, str], None] | None = None,
        early_stopping: bool = True,
    ):
        self.device = torch.device("cpu")
        self.model = model.to(self.device)
        self.learner = learner
        self.image_shape = image_shape
        self.on_batch_end = on_batch_end
        self.early_stopping = early_stopping

    def _get_pre_spike_times(self, input_spike_times: torch.Tensor) -> torch.Tensor:
        """Return pre-synaptic spike times for the learner's layer.

        For a single layer or the first layer in a sequential model, returns
        the input spike times. For deeper layers, returns the previous layer's
        spike times (set during the forward pass).
        """
        if not isinstance(self.model, SpikingSequential):
            return input_spike_times
        try:
            layer_idx = self.model.layers.index(self.learner.layer)
        except ValueError:
            raise ValueError("Learner's layer is not part of the model's layers")
        if layer_idx == 0:
            return input_spike_times
        return self.model.layers[layer_idx - 1].spike_times

    def step_batch(
        self,
        batch_idx: int,
        times: torch.Tensor,
        /,
        split: str = "train",
    ):
        flat_times = times.flatten().to(self.device)
        with torch.no_grad():
            for incoming_spikes, current_time, dt in iterate_spikes(flat_times):
                output_spikes = self.model.forward(incoming_spikes, current_time, dt)
                if self.early_stopping and torch.any(output_spikes == 1.0):
                    break
        dw = 0.0
        if self.model.training:
            pre_spike_times = self._get_pre_spike_times(flat_times)
            dw = self.learner.step(pre_spike_times)

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
