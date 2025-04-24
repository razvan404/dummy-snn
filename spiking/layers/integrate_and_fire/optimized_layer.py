import torch

from spiking.layers.layer import SpikingLayer
from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.threshold import (
    ThresholdInitialization,
    ThresholdAdaptation,
    ConstantInitialization,
)


class IntegrateAndFireOptimizedLayer(SpikingLayer):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
        threshold_initialization: ThresholdInitialization | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        if threshold_initialization is None:
            threshold_initialization = ConstantInitialization()

        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_mechanism=learning_mechanism,
            competition_mechanism=competition_mechanism,
            threshold_initialization=threshold_initialization,
            threshold_adaptation=threshold_adaptation,
        )

        self.device = device
        self.thresholds = threshold_initialization.initialize(
            threshold, shape=(num_outputs,)
        )
        self.refractory_period = refractory_period

        self.membrane_potentials = torch.zeros(
            num_outputs, dtype=torch.float32, device=device
        )
        self.refractory_times = torch.zeros(
            num_outputs, dtype=torch.float32, device=device
        )
        self.weights = torch.rand(
            (num_outputs, num_inputs), dtype=torch.float32, device=device
        )

        self._spike_times = torch.full(
            (num_outputs,), float("inf"), dtype=torch.float32, device=device
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
        input_contrib = torch.sum(self.weights[active_neurons] * incoming_spikes, dim=1)
        self.membrane_potentials[active_neurons] += input_contrib
        spiking_neurons = active_neurons & (self.membrane_potentials >= self.thresholds)

        self.membrane_potentials[spiking_neurons] = 0.0
        unspiked_neurons = spiking_neurons & torch.isinf(self._spike_times)
        self._spike_times[unspiked_neurons] = current_time
        self.refractory_times[spiking_neurons] = self.refractory_period

        return spiking_neurons.float()

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

        total_loss = 0.0
        for neuron_idx in neurons_to_learn:
            idx = neuron_idx.item()
            updated_weights = self.learning_mechanism.update_weights(
                self.weights[idx], pre_spike_times, self._spike_times[idx]
            )
            total_loss += torch.mean(
                torch.abs(self.weights[idx] - updated_weights)
            ).item()
            self.weights[idx] = updated_weights

        if self.threshold_adaptation:
            self.thresholds = self.threshold_adaptation.update(
                self.thresholds,
                self._spike_times,
                neurons_to_learn=neurons_to_learn,
            )

        return total_loss / len(neurons_to_learn) if len(neurons_to_learn) > 0 else 0.0

    def reset(self):
        self.membrane_potentials.zero_()
        self.refractory_times.zero_()
        self._spike_times.fill_(float("inf"))

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times
