import torch

from .adaptation import ThresholdAdaptation


class WeightMeanAdaptation(ThresholdAdaptation):
    """Drive per-filter mean weight toward ``target_mean``.

    Per-neuron mode (``use_winner=False``):
        ``Δθ_i = lr · (target_mean − mean(W_i))``
        Each neuron is independently pushed toward the target.

    Global mode (``use_winner=True``):
        ``Δθ = lr · (target_mean − mean(W_winner))`` applied to **all**
        thresholds.
    """

    def __init__(
        self,
        min_threshold: float,
        learning_rate: float,
        target_mean: float = 0.5,
        decay_factor: float = 1.0,
        max_threshold: float | None = None,
        use_winner: bool = False,
    ):
        super().__init__()
        self.min_threshold = float(min_threshold)
        self.learning_rate = float(learning_rate)
        self.target_mean = float(target_mean)
        self.decay_factor = float(decay_factor)
        self.max_threshold = float(max_threshold) if max_threshold is not None else None
        self.use_winner = bool(use_winner)

    def learning_rate_step(self):
        self.learning_rate *= self.decay_factor

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        weights = kwargs.get("weights")
        if weights is None:
            raise ValueError("`weights` must be provided.")
        flat_w = weights.flatten(1)
        if self.use_winner:
            winners = kwargs.get("neurons_to_learn")
            if winners is None or len(winners) == 0:
                return current_thresholds.clone()
            winner_mean = flat_w[winners].mean()
            delta = self.learning_rate * (self.target_mean - winner_mean)
        else:
            mean_per_neuron = flat_w.mean(dim=1)
            delta = self.learning_rate * (self.target_mean - mean_per_neuron)
        return torch.clamp(
            current_thresholds + delta,
            min=self.min_threshold,
            max=self.max_threshold,
        )
