import torch

from .competition import CompetitionMechanism
from spiking.learning.mechanism import LearningMechanism
from spiking.spiking_module import SpikingModule
from spiking.threshold import ThresholdAdaptation


class Learner:
    def __init__(
        self,
        layer: SpikingModule,
        learning_mechanism: LearningMechanism | None = None,
        competition: CompetitionMechanism | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
    ):
        self.layer = layer
        self.learning_mechanism = learning_mechanism
        self.competition = competition
        self.threshold_adaptation = threshold_adaptation

    def _select_neurons(self) -> torch.Tensor:
        if self.competition:
            return self.competition.neurons_to_learn(self.layer.spike_times)
        return torch.nonzero(torch.isfinite(self.layer.spike_times), as_tuple=False)

    @torch.no_grad()
    def step(self, pre_spike_times: torch.Tensor) -> float:
        neurons_to_learn = self._select_neurons()
        self.neurons_to_learn = neurons_to_learn

        total_dw = 0.0
        if self.learning_mechanism:
            for neuron_idx in neurons_to_learn:
                idx = neuron_idx.item()
                updated_weights = self.learning_mechanism.update_weights(
                    self.layer.weights[idx],
                    pre_spike_times,
                    self.layer.spike_times[idx],
                )
                total_dw += torch.mean(
                    torch.abs(self.layer.weights[idx] - updated_weights)
                ).item()
                if self.layer.training:
                    self.layer.weights[idx].copy_(updated_weights)

        if self.threshold_adaptation and self.layer.training:
            self.layer.thresholds.copy_(
                self.threshold_adaptation.update(
                    self.layer.thresholds,
                    self.layer.spike_times,
                    neurons_to_learn=neurons_to_learn,
                    weights=self.layer.weights,
                    pre_spike_times=pre_spike_times,
                )
            )

        if not self.learning_mechanism:
            return 0.0
        return total_dw / len(neurons_to_learn) if len(neurons_to_learn) > 0 else 0.0

    def learning_rate_step(self):
        if self.learning_mechanism:
            self.learning_mechanism.learning_rate_step()
        if self.threshold_adaptation:
            self.threshold_adaptation.learning_rate_step()
