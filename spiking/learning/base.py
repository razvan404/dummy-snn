from abc import ABC, abstractmethod

import torch

from .competition import CompetitionMechanism
from .mechanism import LearningMechanism
from spiking.spiking_module import SpikingModule
from spiking.threshold import ThresholdAdaptation


class BaseLearner(ABC):
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

    def _get_spike_times(self) -> torch.Tensor:
        return self.layer.spike_times

    def _select_neurons(self) -> torch.Tensor:
        spike_times = self._get_spike_times()
        if self.competition:
            return self.competition.neurons_to_learn(spike_times)
        return torch.nonzero(torch.isfinite(spike_times), as_tuple=False).flatten()

    @abstractmethod
    def _update_weights(
        self, neurons_to_learn: torch.Tensor, pre_spike_times: torch.Tensor
    ) -> float:
        """Returns avg |dw|."""

    @torch.no_grad()
    def step(self, pre_spike_times: torch.Tensor) -> torch.Tensor:
        """Caller may defer host sync of returned tensor."""
        neurons_to_learn = self._select_neurons().flatten()
        self.neurons_to_learn = neurons_to_learn
        spike_times_now = self._get_spike_times()
        device = spike_times_now.device
        if len(neurons_to_learn) > 0:
            self.winner_spike_time = spike_times_now[neurons_to_learn[0]].min()
        else:
            self.winner_spike_time = torch.tensor(float("inf"), device=device)

        dw = torch.zeros((), device=device, dtype=spike_times_now.dtype)
        if self.learning_mechanism and len(neurons_to_learn) > 0:
            dw = self._update_weights(neurons_to_learn, pre_spike_times)

        if self.threshold_adaptation and self.layer.training:
            spike_times = self._get_spike_times()
            self.layer.thresholds.copy_(
                self.threshold_adaptation.update(
                    self.layer.thresholds,
                    spike_times,
                    neurons_to_learn=neurons_to_learn,
                    weights=self.layer.weights,
                    pre_spike_times=pre_spike_times,
                )
            )

        if not self.learning_mechanism:
            return torch.zeros((), device=device, dtype=spike_times_now.dtype)
        return dw

    def learning_rate_step(self):
        if self.learning_mechanism:
            self.learning_mechanism.learning_rate_step()
        if self.threshold_adaptation:
            self.threshold_adaptation.learning_rate_step()
