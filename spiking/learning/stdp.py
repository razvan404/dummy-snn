import torch
from spiking.learning.mechanism import LearningMechanism
from spiking.registry import registry


@registry.register("learning_mechanism", "stdp")
class STDP(LearningMechanism):
    def __init__(
        self,
        tau_pre: float = 20,
        tau_post: float = 20,
        learning_rate: float = 0.01,
        decay_factor: float = 1.0,
        max_pre_spike_time: float = None,
        weights_interval: tuple[float, float] = (0, 1),
    ):
        if max_pre_spike_time is None:
            raise ValueError("`max_pre_spike_time` must be provided.")
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.max_pre_spike_time = max_pre_spike_time
        self.weights_interval = weights_interval

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update_weights(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_times: torch.Tensor,
    ) -> torch.Tensor:
        pre_spike_times = torch.clamp(pre_spike_times, max=self.max_pre_spike_time)
        delta_t = post_spike_times - pre_spike_times

        dw = torch.zeros_like(weights)

        potentiation_mask = delta_t > 0
        depression_mask = delta_t < 0

        dw[potentiation_mask] = torch.exp(-delta_t[potentiation_mask] / self.tau_pre)
        dw[depression_mask] = -torch.exp(delta_t[depression_mask] / self.tau_post)

        updated_weights = weights + self.learning_rate * dw
        return torch.clamp(updated_weights, *self.weights_interval)
