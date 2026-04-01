import torch

from .adaptation import ThresholdAdaptation


class CompetitiveThresholdAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = float(min_threshold)
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
        N = len(current_thresholds)

        losers_mask = torch.ones(
            N, dtype=torch.bool, device=current_thresholds.device
        )
        losers_mask[winners_neurons] = False

        # Paper 19 Eq 9: winner gets +eta_th, losers get -eta_th/N
        threshold_updates = torch.zeros_like(current_thresholds)
        if len(winners_neurons) > 0:
            threshold_updates[winners_neurons] = self.learning_rate
        if losers_mask.any():
            threshold_updates[losers_mask] = -self.learning_rate / N

        return torch.clamp(
            current_thresholds + threshold_updates, min=self.min_threshold
        )
