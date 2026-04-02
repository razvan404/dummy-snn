import torch

from .adaptation import ThresholdAdaptation


class TargetTimestampAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        target_timestamp: float | torch.Tensor,
        learning_rate: float,
        decay_factor: float = 1.0,
        epsilon: float = 0.0,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.learning_rate = learning_rate
        self.target_timestamp = target_timestamp
        self.decay_factor = decay_factor
        self.epsilon = epsilon

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Update thresholds toward the target timestamp (Falez 2020 Eq 6).

        When ``neurons_to_learn`` (winners) is provided, uses the winner's
        spike time and applies the update to **all** neurons (paper Eq 6:
        "each time a winning neuron fires a spike, all the neurons in
        competition apply the update").

        Without ``neurons_to_learn``, falls back to per-neuron updates using
        each neuron's own spike time.
        """
        # Resolve target timestamp to a scalar value.
        if isinstance(self.target_timestamp, torch.Tensor):
            target_val = self.target_timestamp.to(current_thresholds.device)
        else:
            target_val = float(self.target_timestamp)

        neurons_to_learn = kwargs.get("neurons_to_learn")
        if neurons_to_learn is not None and len(neurons_to_learn) > 0:
            # Paper Eq 6: use the winner's spike time for ALL neurons.
            winner_time = spike_times[neurons_to_learn].min()
            if not torch.isfinite(winner_time):
                return current_thresholds
            error = winner_time - target_val  # scalar
            if self.epsilon > 0 and abs(error) <= self.epsilon:
                return current_thresholds
            if self.epsilon > 0:
                error = error - error.sign() * self.epsilon
            threshold_delta = self.learning_rate * error
            updated_thresholds = current_thresholds - threshold_delta
        else:
            # Fallback: per-neuron updates using each neuron's own spike time.
            threshold_delta = torch.zeros_like(current_thresholds)
            if isinstance(target_val, float):
                target = torch.full_like(current_thresholds, target_val)
            else:
                target = target_val
            finite_mask = torch.isfinite(spike_times)
            err = spike_times[finite_mask] - target[finite_mask]
            if self.epsilon > 0:
                within = err.abs() <= self.epsilon
                err = torch.where(
                    within, torch.zeros_like(err), err - torch.sign(err) * self.epsilon
                )
            threshold_delta[finite_mask] = self.learning_rate * err
            updated_thresholds = current_thresholds - threshold_delta

        return torch.clamp(updated_thresholds, min=self.min_threshold)
