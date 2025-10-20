import torch
import torch.nn as nn

from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.threshold import (
    ThresholdInitialization,
    ThresholdAdaptation,
)

from ..layer import SpikingLayer
from ..surrogate_spike import SurrogateSpike


class IntegrateAndFireLayer(SpikingLayer):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        threshold_initialization: ThresholdInitialization,
        competition_mechanism: CompetitionMechanism | None = None,
        refractory_period: float = 1.0,
        threshold_adaptation: ThresholdAdaptation | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_mechanism=learning_mechanism,
            competition_mechanism=competition_mechanism,
            threshold_initialization=threshold_initialization,
            threshold_adaptation=threshold_adaptation,
        )

        self.refractory_period = refractory_period

        self.weights = nn.Parameter(torch.rand((num_outputs, num_inputs), dtype=dtype))
        self.thresholds = nn.Parameter(
            threshold_initialization.initialize((num_outputs,))
        )

        self.register_buffer(
            "membrane_potentials", torch.zeros((num_outputs,), dtype=dtype)
        )
        self.register_buffer(
            "refractory_times", torch.zeros((num_outputs,), dtype=dtype)
        )
        self.register_buffer(
            "_spike_times", torch.full((num_outputs,), float("inf"), dtype=dtype)
        )

    def _update_refractory(self, dt: float) -> torch.Tensor:
        active_neurons = self.refractory_times == 0
        self.refractory_times[~active_neurons] = torch.clamp(
            self.refractory_times[~active_neurons] - dt, min=0.0
        )
        return active_neurons

    def _update_potential(
        self,
        incoming_spikes: torch.Tensor,
        current_time: float,
        active_neurons: torch.Tensor,
    ) -> torch.Tensor:
        surrogate_spikes = torch.zeros_like(self.membrane_potentials)

        if not active_neurons.any():
            return surrogate_spikes

        input_contrib = torch.sum(self.weights[active_neurons] * incoming_spikes, dim=1)
        self.membrane_potentials[active_neurons] += input_contrib

        potentials = self.membrane_potentials[active_neurons]
        thresholds = self.thresholds[active_neurons]

        spikes_active = SurrogateSpike.apply(potentials, thresholds)
        surrogate_spikes[active_neurons] = spikes_active

        spiking_mask_full = surrogate_spikes > 0.0
        self.membrane_potentials[spiking_mask_full] = 0.0

        unspiked_neurons = spiking_mask_full & torch.isinf(self._spike_times)
        self._spike_times[unspiked_neurons] = current_time
        self.refractory_times[spiking_mask_full] = self.refractory_period

        return surrogate_spikes

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        active_neurons = self._update_refractory(dt)
        return self._update_potential(incoming_spikes, current_time, active_neurons)

    def backward(self, pre_spike_times: torch.Tensor) -> float:
        neurons_to_learn = (
            self.competition_mechanism.neurons_to_learn(self._spike_times)
            if self.competition_mechanism
            else torch.nonzero(torch.isfinite(self._spike_times), as_tuple=False)
        )

        total_dw = 0.0
        for neuron_idx in neurons_to_learn:
            idx = neuron_idx.item()
            updated_weights = self.learning_mechanism.update_weights(
                self.weights[idx], pre_spike_times, self._spike_times[idx]
            )
            total_dw += torch.mean(
                torch.abs(self.weights[idx] - updated_weights)
            ).item()
            if self.training:
                self.weights[idx].copy_(updated_weights)

        if self.threshold_adaptation and self.training:
            self.thresholds.copy_(
                self.threshold_adaptation.update(
                    self.thresholds,
                    self._spike_times,
                    neurons_to_learn=neurons_to_learn,
                )
            )

        return total_dw / len(neurons_to_learn) if len(neurons_to_learn) > 0 else 0.0

    def reset(self):
        self.membrane_potentials.zero_()
        self.refractory_times.zero_()
        self._spike_times.fill_(float("inf"))

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times
