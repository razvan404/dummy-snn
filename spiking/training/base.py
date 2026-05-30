from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from spiking.learning.base import BaseLearner
from spiking.spiking_module import SpikingModule
from spiking.layers.sequential import SpikingSequential


class BaseUnsupervisedTrainer(ABC):
    def __init__(
        self,
        model: SpikingModule,
        learner: BaseLearner,
        image_shape: tuple[int, int, int],
        on_batch_end: Callable[[int, float, str], None] | None = None,
        early_stopping: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.learner = learner
        self.image_shape = image_shape
        self.on_batch_end = on_batch_end
        self.early_stopping = early_stopping

    @abstractmethod
    def _prepare_input(self, times: torch.Tensor) -> torch.Tensor:
        ...

    def _get_pre_spike_times(self, prepared_times: torch.Tensor) -> torch.Tensor:
        return prepared_times

    def _write_spike_times(
        self, layer: SpikingModule, spike_times: torch.Tensor
    ) -> None:
        if hasattr(layer, "_oH") and layer._oH is None:
            if spike_times.dim() >= 3:
                oH, oW = spike_times.shape[-2], spike_times.shape[-1]
                layer._init_spatial_buffers(oH, oW)
        layer._spike_times.copy_(spike_times)

    def _forward_analytical(self, prepared: torch.Tensor) -> None:
        if isinstance(self.model, SpikingSequential):
            times = prepared
            for layer in self.model.layers:
                spike_times = layer.infer_spike_times(times)
                self._write_spike_times(layer, spike_times)
                times = spike_times
        else:
            spike_times = self.model.infer_spike_times(prepared)
            self._write_spike_times(self.model, spike_times)

    def step_batch(
        self,
        batch_idx: int,
        times: torch.Tensor,
        /,
        split: str = "train",
    ):
        prepared = self._prepare_input(times)
        with torch.no_grad():
            self._forward_analytical(prepared)
        dw = 0.0
        if self.model.training:
            pre_spike_times = self._get_pre_spike_times(prepared)
            dw = self.learner.step(pre_spike_times)

        if self.on_batch_end:
            self.on_batch_end(batch_idx, dw, split)
        self.model.reset()
        return dw

    def step_loader(
        self,
        loader: DataLoader,
        /,
        split: str = "train",
        progress: bool = False,
    ):
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        if hasattr(loader, "dataset"):
            self._step_loader_direct(
                loader.dataset, split, progress, shuffle=(split == "train")
            )
        else:
            self._step_loader_iterable(loader, split, progress)

    def _step_loader_direct(
        self,
        dataset,
        split: str,
        progress: bool,
        shuffle: bool,
    ):
        n = len(dataset)
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        it = range(n)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=n, desc=split, unit="sample", leave=False)
        for batch_idx in it:
            times, _ = dataset[indices[batch_idx]]
            self.step_batch(batch_idx, times, split=split)

    def _step_loader_iterable(self, loader, split: str, progress: bool):
        it = enumerate(loader)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(loader), desc=split, unit="sample", leave=False)
        for batch_idx, (times, _label) in it:
            self.step_batch(batch_idx, times, split=split)

    def step_epoch(self):
        self.learner.learning_rate_step()
