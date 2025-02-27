import numpy as np

from .adaptation import ThresholdAdaptation


class FalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        threshold_learning_rate: float,
        target_timestamp: float,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.threshold_learning_rate = threshold_learning_rate
        self.target_timestamp = target_timestamp

    def update(
        self, current_thresholds: np.ndarray, spike_times: np.ndarray
    ) -> np.ndarray:
        threshold_delta = np.zeros_like(current_thresholds)
        threshold_delta[np.isfinite(spike_times)] = self.threshold_learning_rate * (
            spike_times[np.isfinite(spike_times)] - self.target_timestamp
        )
        return current_thresholds - threshold_delta
