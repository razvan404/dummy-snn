import torch
from .adaptation import ThresholdAdaptation


class FalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        target_timestamp: float,
        learning_rate: float,
        decay_factor: float = 1.0,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.learning_rate = learning_rate
        self.target_timestamp = target_timestamp
        self.decay_factor = decay_factor

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        threshold_delta = torch.zeros_like(current_thresholds)

        finite_mask = torch.isfinite(spike_times)
        threshold_delta[finite_mask] = self.learning_rate * (
            spike_times[finite_mask] - self.target_timestamp
        )

        updated_thresholds = current_thresholds - threshold_delta
        return torch.maximum(
            updated_thresholds,
            torch.tensor(self.min_threshold, device=current_thresholds.device),
        )
