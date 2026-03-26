import torch

from .base import BaseUnsupervisedTrainer
from spiking.layers.sequential import SpikingSequential


class UnsupervisedTrainer(BaseUnsupervisedTrainer):
    """Trainer for fully-connected (non-spatial) spiking layers.

    Flattens input spike times and supports multilayer models by
    looking up the correct pre-synaptic spike times.
    """

    def _prepare_input(self, times: torch.Tensor) -> torch.Tensor:
        return times.flatten().to(self.device)

    def _get_pre_spike_times(self, input_spike_times: torch.Tensor) -> torch.Tensor:
        """Return pre-synaptic spike times for the learner's layer.

        For a single layer or the first layer in a sequential model, returns
        the input spike times. For deeper layers, returns the previous layer's
        spike times (set during the forward pass).
        """
        if not isinstance(self.model, SpikingSequential):
            return input_spike_times
        try:
            layer_idx = next(
                i for i, l in enumerate(self.model.layers) if l is self.learner.layer
            )
        except StopIteration:
            raise ValueError("Learner's layer is not part of the model's layers")
        if layer_idx == 0:
            return input_spike_times
        return self.model.layers[layer_idx - 1].spike_times
