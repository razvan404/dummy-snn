import numpy as np

from .adaptation import ThresholdAdaptation


class FalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        target_timestamp: float,
        threshold_learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.learning_rate = threshold_learning_rate
        self.target_timestamp = target_timestamp
        self.decay_factor = decay_factor

    def _learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray:
        threshold_delta = np.zeros_like(current_thresholds)
        threshold_delta[np.isfinite(spike_times)] = self.learning_rate * (
            spike_times[np.isfinite(spike_times)] - self.target_timestamp
        )
        self._learning_rate_step()
        return current_thresholds - threshold_delta
