import numpy as np

from spiking.learning.mechanism import LearningMechanism


class STDP(LearningMechanism):
    __slots__ = ["tau_pre", "tau_post", "lr"]

    def __init__(self, tau_pre: float = 20, tau_post: float = 20, lr: float = 0.01):
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

    def update_weights(
        self,
        weights: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray,
    ) -> np.ndarray:
        delta_t = post_spike_times - pre_spike_times
        dw = np.zeros_like(weights)

        dw[delta_t > 0] = self.lr * np.exp(-delta_t[delta_t > 0] / self.tau_pre)
        dw[delta_t < 0] = -self.lr * np.exp(delta_t[delta_t < 0] / self.tau_post)

        return np.clip(weights + dw, 0, 1)
