import numpy as np

from .adaptation import ThresholdAdaptation


class CompetitiveFalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray:
        winner_index = np.argmin(spike_times)
        # TODO: check for winner infinity, also make this random

        threshold_updates = np.ones(len(current_thresholds)) * (
            -self.learning_rate / len(spike_times)
        )
        threshold_updates[winner_index] = self.learning_rate
        updated_thresholds = np.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )

        return updated_thresholds
