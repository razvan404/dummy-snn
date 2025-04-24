import torch
from .adaptation import ThresholdAdaptation


class CompetitiveFalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = torch.tensor(min_threshold)
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if (neurons_to_learn := kwargs.get("neurons_to_learn")) is None:
            raise ValueError("`neurons_to_learn` must be provided.")

        winners_neurons = neurons_to_learn
        winners_divisor = len(winners_neurons)

        all_indices = torch.arange(len(current_thresholds))
        losers_neurons = all_indices[~torch.isin(all_indices, winners_neurons)]
        losers_divisor = len(losers_neurons)

        threshold_updates = torch.ones_like(current_thresholds)
        if winners_divisor > 0:
            threshold_updates[winners_neurons] *= self.learning_rate / winners_divisor
        if losers_divisor > 0:
            threshold_updates[losers_neurons] *= -self.learning_rate / losers_divisor

        updated_thresholds = torch.maximum(
            self.min_threshold, current_thresholds + threshold_updates
        )
        return updated_thresholds
