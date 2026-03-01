import torch

from .adaptation import ThresholdAdaptation


class PlasticityBalanceAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        tau: float = 20.0,
        learning_rate: float = 0.1,
        decay_factor: float = 1.0,
        min_threshold: float = 1.0,
        max_threshold: float = 100.0,
        sign_only: bool = False,
    ):
        self.tau = tau
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.sign_only = sign_only

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def compute_balance(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_time: float,
    ) -> float:
        """
        Each synapse contributes weight * exp(-|delta_t|/tau), with sign
        determined by whether the pre-synaptic spike preceded (potentiation)
        or followed (depression) the post-synaptic spike.
        """
        delta_t = post_spike_time - pre_spike_times

        potentiation_mask = delta_t > 0
        depression_mask = delta_t < 0

        pot_contribution = (
            weights[potentiation_mask]
            * torch.exp(-delta_t[potentiation_mask] / self.tau)
        ).sum()

        dep_contribution = (
            weights[depression_mask] * torch.exp(delta_t[depression_mask] / self.tau)
        ).sum()

        return (pot_contribution - dep_contribution).item()

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        weights = kwargs["weights"]
        pre_spike_times = kwargs["pre_spike_times"]

        threshold_deltas = torch.zeros_like(current_thresholds)

        spiked_mask = torch.isfinite(spike_times)
        spiked_indices = torch.nonzero(spiked_mask, as_tuple=True)[0]

        for neuron_idx in spiked_indices:
            idx = neuron_idx.item()
            post_spike_time = spike_times[idx].item()
            neuron_weights = weights[idx]

            balance = self.compute_balance(
                neuron_weights, pre_spike_times, post_spike_time
            )

            if self.sign_only:
                direction = 1.0 if balance > 0 else (-1.0 if balance < 0 else 0.0)
                threshold_deltas[idx] = self.learning_rate * direction
            else:
                threshold_deltas[idx] = self.learning_rate * balance

        new_thresholds = current_thresholds + threshold_deltas
        return torch.clamp(new_thresholds, self.min_threshold, self.max_threshold)
