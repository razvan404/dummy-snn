from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from spiking.learning.base import BaseLearner
from spiking.spiking_module import SpikingModule
from spiking.layers.sequential import SpikingSequential


class BaseUnsupervisedTrainer(ABC):
    """Base trainer for unsupervised spiking networks.

    Subclasses must implement:
        _prepare_input: how to prepare raw input times for the forward pass.
    Optionally override:
        _get_pre_spike_times: how to derive pre-synaptic spike times for learning.
    """

    def __init__(
        self,
        model: SpikingModule,
        learner: BaseLearner,
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

    @abstractmethod
    def _prepare_input(self, times: torch.Tensor) -> torch.Tensor:
        """Prepare raw input times for the forward pass."""

    def _get_pre_spike_times(self, prepared_times: torch.Tensor) -> torch.Tensor:
        """Get pre-synaptic spike times for the learner.

        Default: return the prepared input times.
        Override for multilayer support.
        """
        return prepared_times

    def _write_spike_times(self, layer: SpikingModule, spike_times: torch.Tensor) -> None:
        """Write inferred spike times into the layer's state buffer.

        Handles lazy spatial buffer initialization for conv layers whose
        _spike_times buffer doesn't exist until the first forward() call.
        """
        if hasattr(layer, "_spatial_initialized") and not layer._spatial_initialized:
            oH, oW = spike_times.shape[1], spike_times.shape[2]
            layer._ensure_spatial_buffers(oH, oW)
        layer._spike_times.copy_(spike_times)

    def _forward_analytical(self, prepared: torch.Tensor) -> None:
        """Run analytical spike time inference and write results into model state.

        For SpikingSequential, writes each layer's spike times so that
        _get_pre_spike_times can find the previous layer's output.
        """
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
        """Fast path: direct dataset indexing, bypasses DataLoader overhead."""
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
        """Fallback: iterate over any iterable (DataLoader or list)."""
        it = enumerate(loader)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(loader), desc=split, unit="sample", leave=False)
        for batch_idx, (times, _label) in it:
            self.step_batch(batch_idx, times, split=split)

    def step_epoch(self):
        self.learner.learning_rate_step()
