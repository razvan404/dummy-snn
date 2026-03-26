import torch

from .base import BaseLearner


class Learner(BaseLearner):
    """Learner for fully-connected (non-spatial) spiking layers."""

    def _update_weights(
        self, neurons_to_learn: torch.Tensor, pre_spike_times: torch.Tensor
    ) -> float:
        weights_slice = self.layer.weights[neurons_to_learn]
        post_spike_times = self.layer.spike_times[neurons_to_learn].unsqueeze(1)
        updated_weights = self.learning_mechanism.update_weights(
            weights_slice,
            pre_spike_times,
            post_spike_times,
        )
        dw = torch.mean(torch.abs(weights_slice - updated_weights)).item()
        if self.layer.training:
            self.layer.weights.data[neurons_to_learn] = updated_weights
        return dw
