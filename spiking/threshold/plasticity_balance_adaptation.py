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
        homeostatic: bool = False,
    ):
        self.tau = tau
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.sign_only = sign_only
        self.homeostatic = homeostatic

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
        if not spiked_mask.any():
            return current_thresholds.clone()

        spiked_post = spike_times[spiked_mask].unsqueeze(1)  # (K, 1)
        spiked_w = weights[spiked_mask]  # (K, N)
        delta_t = spiked_post - pre_spike_times.unsqueeze(0)  # (K, N)

        potentiation_mask = delta_t > 0
        depression_mask = delta_t < 0
        exp_decay = torch.exp(-delta_t.abs() / self.tau)

        pot = (spiked_w * potentiation_mask * exp_decay).sum(dim=1)
        dep = (spiked_w * depression_mask * exp_decay).sum(dim=1)
        balance = pot - dep

        if self.sign_only:
            threshold_deltas[spiked_mask] = self.learning_rate * balance.sign()
        else:
            threshold_deltas[spiked_mask] = self.learning_rate * balance

        if self.homeostatic:
            threshold_deltas = -threshold_deltas

        new_thresholds = current_thresholds - threshold_deltas
        return torch.clamp(new_thresholds, self.min_threshold, self.max_threshold)
