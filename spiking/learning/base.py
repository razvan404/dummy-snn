from abc import ABC, abstractmethod

import torch

from .competition import CompetitionMechanism
from .mechanism import LearningMechanism
from spiking.spiking_module import SpikingModule
from spiking.threshold import ThresholdAdaptation


class BaseLearner(ABC):
    """Orchestrates STDP learning, competition, and threshold adaptation.

    Subclasses must implement:
        _update_weights: apply the learning rule to selected neurons.
    Optionally override:
        _get_spike_times: how to derive spike times for competition/adaptation.
    """

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
        """Get spike times for competition and threshold adaptation."""
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
        """Apply learning rule to selected neurons.

        :param neurons_to_learn: indices of neurons to update.
        :param pre_spike_times: pre-synaptic spike times.
        :returns: Average absolute weight change.
        """

    @torch.no_grad()
    def step(self, pre_spike_times: torch.Tensor) -> float:
        """Apply one learning step after a forward pass.

        :param pre_spike_times: pre-synaptic spike times.
        :returns: Average absolute weight change.
        """
        neurons_to_learn = self._select_neurons().flatten()
        self.neurons_to_learn = neurons_to_learn
        # Capture spike times before reset clears layer state
        spike_times_now = self._get_spike_times()
        if len(neurons_to_learn) > 0:
            self.winner_spike_time = spike_times_now[neurons_to_learn[0]].min().item()
        else:
            self.winner_spike_time = float("inf")

        dw = 0.0
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
            return 0.0
        return dw

    def learning_rate_step(self):
        if self.learning_mechanism:
            self.learning_mechanism.learning_rate_step()
        if self.threshold_adaptation:
            self.threshold_adaptation.learning_rate_step()
