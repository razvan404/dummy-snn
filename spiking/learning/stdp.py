import numpy as np

from spiking.learning.mechanism import LearningMechanism


class STDP(LearningMechanism):
    __slots__ = ["tau_pre", "tau_post", "lr"]

    def __init__(
        self,
        tau_pre: float = 20,
        tau_post: float = 20,
        lr: float = 0.01,
        decay_factor: float = 1.0,
    ):
        """
        Spike-Timing-Dependent Plasticity (STDP).

        Args:
            tau_pre (float): Time constant for pre-synaptic spike.
            tau_post (float): Time constant for post-synaptic spike.
            lr (float): Learning rate for weight updates.
        """
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr = lr
        self.decay_factor = decay_factor

    def _learning_rate_step(self):
        self.lr *= self.decay_factor

    def update_weights(
        self,
        weights: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
    ) -> np.ndarray:
        delta_t = post_spike_times - pre_spike_times
        finite_mask = np.isfinite(delta_t)

        dw = np.zeros_like(weights)
        potentiation = finite_mask & (delta_t > 0)
        depression = finite_mask & (delta_t < 0)

        dw[potentiation] = np.exp(-delta_t[potentiation] / self.tau_pre)
        dw[depression] = -np.exp(delta_t[depression] / self.tau_post)

        updated_weights = np.clip(weights + self.lr * dw, 0, 1)
        self._learning_rate_step()
        return updated_weights
