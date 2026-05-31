import torch

from spikinn.learning.mechanism import LearningMechanism


class MultiplicativeSTDP(LearningMechanism):
    def __init__(
        self,
        learning_rate: float,
        decay_factor: float = 1.0,
        beta: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        t_ltp: float = float("inf"),
    ):
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.beta = beta
        self.w_min = w_min
        self.w_max = w_max
        self.t_ltp = t_ltp

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update_weights(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_times: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = post_spike_times - pre_spike_times
        w_range = self.w_max - self.w_min

        potentiate = (delta_t > 0) & (delta_t <= self.t_ltp)

        pot_scale = torch.exp(-self.beta * (weights - self.w_min) / w_range)
        dep_scale = torch.exp(-self.beta * (self.w_max - weights) / w_range)

        dw = torch.where(
            potentiate,
            self.learning_rate * pot_scale,
            -self.learning_rate * dep_scale,
        )

        updated = weights + dw
        return torch.clamp(updated, self.w_min, self.w_max)
