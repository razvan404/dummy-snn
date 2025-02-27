import numpy as np

from .adaptation import ThresholdAdaptation


class CompetitiveFalezAdaptation(ThresholdAdaptation):
    def __init__(self, min_threshold: float, threshold_learning_rate: float):
        super().__init__()
        self.min_threshold = min_threshold
        self.threshold_learning_rate = threshold_learning_rate

    def update(
        self, current_thresholds: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray:
        winner_index = np.argmin(spike_times)

        threshold_updates = np.ones(len(current_thresholds)) * (
            -self.threshold_learning_rate / len(spike_times)
        )
        threshold_updates[winner_index] = self.threshold_learning_rate

        updated_thresholds = np.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )

        return updated_thresholds
