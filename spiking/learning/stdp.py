import numpy as np

from spiking.learning.mechanism import LearningMechanism


class STDP(LearningMechanism):
    __slots__ = [
        "tau_pre",
        "tau_post",
        "lr",
        "decay_factor",
        "max_pre_spike_time",
        "weights_interval",
    ]

    def __init__(
        self,
        tau_pre: float = 20,
        tau_post: float = 20,
        lr: float = 0.01,
        decay_factor: float = 1.0,
        max_pre_spike_time: float = None,
        weights_interval: (float, float) = (0, 1),
    ):
        """
        Spike-Timing-Dependent Plasticity (STDP).
        """
        if max_pre_spike_time is None:
            raise ValueError("`max_pre_spike_time` must be provided.")
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr = lr
        self.decay_factor = decay_factor
        self.max_pre_spike_time = max_pre_spike_time
        self.weights_interval = weights_interval

    def learning_rate_step(self):
        self.lr *= self.decay_factor

    def update_weights(
        self,
        weights: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
    ) -> np.ndarray:
        pre_spike_times[pre_spike_times > self.max_pre_spike_time] = (
            self.max_pre_spike_time
        )
        delta_t = post_spike_times - pre_spike_times

        dw = np.zeros_like(weights)
        potentiation = delta_t > 0
        depression = delta_t < 0

        dw[potentiation] = np.exp(-delta_t[potentiation] / self.tau_pre)
        dw[depression] = -np.exp(delta_t[depression] / self.tau_post)

        updated_weights = np.clip(weights + self.lr * dw, *self.weights_interval)
        return updated_weights
