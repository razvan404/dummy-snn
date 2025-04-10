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
        self, current_thresholds: np.ndarray, spike_times: np.ndarray, **kwargs
    ) -> np.ndarray:
        if (neurons_to_learn := kwargs.get("neurons_to_learn")) is None:
            raise ValueError("`neurons_to_learn` must be provided.")

        winners_neurons = neurons_to_learn
        winners_divisor = len(winners_neurons)

        losers_neurons = np.setdiff1d(
            np.arange(len(current_thresholds)), winners_neurons
        )
        losers_divisor = len(losers_neurons)

        threshold_updates = np.ones(len(current_thresholds))
        if len(winners_neurons) > 0:
            threshold_updates[winners_neurons] *= self.learning_rate / winners_divisor
        if len(losers_neurons) > 0:
            threshold_updates[losers_neurons] *= -self.learning_rate / losers_divisor

        updated_thresholds = np.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )
        return updated_thresholds
