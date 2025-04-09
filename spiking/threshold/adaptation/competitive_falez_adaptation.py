import numpy as np

from .adaptation import ThresholdAdaptation
from ...utils import choose_random_winner


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
        self, current_thresholds: np.ndarray, spike_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        if (neurons_to_learn := kwargs.get("neurons_to_learn")) is None:
            raise ValueError("`neurons_to_learn` must be provided.")
        winner_index = choose_random_winner(spike_times[neurons_to_learn])

        # N - 1 to make the sum of the differences equal to 0
        threshold_updates = np.ones(len(current_thresholds)) * (
            -self.learning_rate / (len(current_thresholds) - 1)
        )
        if winner_index is not None:
            threshold_updates[winner_index] = self.learning_rate
        updated_thresholds = np.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )

        return updated_thresholds
