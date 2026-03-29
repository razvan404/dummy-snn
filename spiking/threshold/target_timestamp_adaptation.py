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
        """Update thresholds with optional dead‑zone (epsilon).

        * ``current_thresholds`` – tensor of shape ``[n_neurons]``.
        * ``spike_times`` – tensor of same shape containing the most recent spike time for each neuron (``nan`` if no spike).
        * ``kwargs`` may contain ``neurons_to_learn`` – a list of indices to restrict the update.
        """
        threshold_delta = torch.zeros_like(current_thresholds)

        # Resolve target timestamps: can be a scalar or a per‑neuron tensor.
        if isinstance(self.target_timestamp, torch.Tensor):
            target = self.target_timestamp.to(current_thresholds.device)
        else:
            target = torch.full_like(current_thresholds, float(self.target_timestamp))

        # Helper to compute dead‑zone adjusted error.
        def compute_error(spike, tgt):
            err = spike - tgt
            if self.epsilon > 0:
                # If within dead‑zone, zero the error.
                within = err.abs() <= self.epsilon
                err = torch.where(within, torch.zeros_like(err), err - torch.sign(err) * self.epsilon)
            return err

        neurons_to_learn = kwargs.get("neurons_to_learn")
        if neurons_to_learn is not None and len(neurons_to_learn) > 0:
            idx = neurons_to_learn
            spike_sel = spike_times[idx]
            finite = torch.isfinite(spike_sel)
            err = compute_error(spike_sel[finite], target[idx][finite])
            threshold_delta[idx][finite] = self.learning_rate * err
        else:
            finite_mask = torch.isfinite(spike_times)
            err = compute_error(spike_times[finite_mask], target[finite_mask])
            threshold_delta[finite_mask] = self.learning_rate * err

        updated_thresholds = current_thresholds - threshold_delta
        return torch.clamp(updated_thresholds, min=self.min_threshold)
