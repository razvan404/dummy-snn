import numpy as np

from .adaptation import ThresholdAdaptation


class CompetitiveFalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        threshold_learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.learning_rate = threshold_learning_rate
        self.decay_factor = decay_factor

    def _learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray:
        winner_index = np.argmin(spike_times)

        threshold_updates = np.ones(len(current_thresholds)) * (
            -self.learning_rate / len(spike_times)
        )
        threshold_updates[winner_index] = self.learning_rate
        updated_thresholds = np.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )

        self._learning_rate_step()
        return updated_thresholds
