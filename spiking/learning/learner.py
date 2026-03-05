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
        return torch.nonzero(torch.isfinite(self.layer.spike_times), as_tuple=False).flatten()

    @torch.no_grad()
    def step(self, pre_spike_times: torch.Tensor) -> float:
        neurons_to_learn = self._select_neurons().flatten()
        self.neurons_to_learn = neurons_to_learn

        dw = 0.0
        if self.learning_mechanism and len(neurons_to_learn) > 0:
            weights_slice = self.layer.weights[neurons_to_learn]
            post_spike_times = self.layer.spike_times[neurons_to_learn].unsqueeze(1)
            updated_weights = self.learning_mechanism.update_weights(
                weights_slice, pre_spike_times, post_spike_times,
            )
            dw = torch.mean(torch.abs(weights_slice - updated_weights)).item()
            if self.layer.training:
                self.layer.weights.data[neurons_to_learn] = updated_weights

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
        return dw

    def learning_rate_step(self):
        if self.learning_mechanism:
            self.learning_mechanism.learning_rate_step()
        if self.threshold_adaptation:
            self.threshold_adaptation.learning_rate_step()
