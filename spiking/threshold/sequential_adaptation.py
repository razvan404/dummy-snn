import torch

from .adaptation import ThresholdAdaptation


class SequentialThresholdAdaptation(ThresholdAdaptation):
    def __init__(self, adaptations: list[ThresholdAdaptation]):
        self.adaptations = adaptations

    def update(
        self, current_thresholds: torch.Tensor, spike_times: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        for adaptation in self.adaptations:
            current_thresholds = adaptation.update(
                current_thresholds, spike_times, **kwargs
            )
        return current_thresholds

    def learning_rate_step(self):
        for adaptation in self.adaptations:
            adaptation.learning_rate_step()
