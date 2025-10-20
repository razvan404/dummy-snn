import torch
from torch.utils.data import DataLoader

from spiking import iterate_spikes, Spike
from spiking.spiking_module import SpikingModule
from .callbacks_interface import CallbacksInterface


class UnsupervisedTrainer:
    def __init__(
        self,
        model: SpikingModule,
        image_shape: (int, int, int),
        callbacks: CallbacksInterface | None = None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.image_shape = image_shape
        self.callbacks = callbacks

    @torch.no_grad()
    def step_batch(
        self,
        batch_idx: int,
        spikes: list[Spike],
        times: torch.Tensor,
        label: str | None,
        /,
        split: str = "train",
    ):
        for incoming_spikes, current_time, dt in iterate_spikes(
            spikes, shape=self.image_shape
        ):
            output_spikes = self.model.forward(
                incoming_spikes.flatten().to(self.device), current_time, dt
            )
            if self.callbacks and self.callbacks.callback_step_spike(
                batch_idx, current_time, output_spikes, split
            ):
                continue
            if torch.any(output_spikes == 1.0):
                break
        pre_spike_times = times.flatten().to(self.device)
        dw = self.model.backward(pre_spike_times)

        if self.callbacks:
            self.callbacks.callback_step(
                batch_idx, pre_spike_times.cpu(), dw, label, split
            )
        self.model.reset()
        return dw

    def step_loader(self, loader: DataLoader, /, split: str = "train"):
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        for batch_idx, (spikes, label, times) in enumerate(loader):
            self.step_batch(batch_idx, spikes, times, label, split=split)

    def step_epoch(self):
        self.model.threshold_adaptation.learning_rate_step()
        self.model.learning_mechanism.learning_rate_step()
