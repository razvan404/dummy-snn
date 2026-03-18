import torch

from spiking.learning.mechanism import LearningMechanism


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
        """Apply multiplicative STDP (Falez 2020 Eq 3).

        Potentiation: pre before post, within t_ltp window.
          dw = +η * exp(-β * (w - w_min) / (w_max - w_min))
        Depression: otherwise.
          dw = -η * exp(-β * (w_max - w) / (w_max - w_min))
        """
        delta_t = post_spike_times - pre_spike_times
        w_range = self.w_max - self.w_min

        # Potentiation: causal (delta_t > 0) AND within t_ltp window
        potentiate = (delta_t > 0) & (delta_t <= self.t_ltp)

        # Weight-dependent scaling
        pot_scale = torch.exp(-self.beta * (weights - self.w_min) / w_range)
        dep_scale = torch.exp(-self.beta * (self.w_max - weights) / w_range)

        dw = torch.where(
            potentiate,
            self.learning_rate * pot_scale,
            -self.learning_rate * dep_scale,
        )

        updated = weights + dw
        return torch.clamp(updated, self.w_min, self.w_max)
