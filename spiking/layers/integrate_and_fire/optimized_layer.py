import numpy as np

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

        self.threshold = threshold_initialization.initialize(
            threshold, shape=(num_outputs,)
        )
        self.threshold_adaptation = threshold_adaptation

        self.membrane_potentials = np.zeros(num_outputs, dtype=np.float32)
        self.refractory_times = np.zeros(num_outputs, dtype=np.float32)
        self.weights = np.abs(np.random.rand(num_outputs, num_inputs)) * 0.1
        self.refractory_period = refractory_period

        self._spike_times = np.ones(num_outputs, dtype=np.float32) * np.inf

    def _update_refractory(self, dt: float):
        active_neurons = self.refractory_times == 0
        self.refractory_times = np.maximum(self.refractory_times - dt, 0)
        return active_neurons

    def _update_potential(
        self,
        incoming_spikes: np.ndarray,
        current_time: float,
        active_neurons: np.ndarray,
    ):
        self.membrane_potentials[active_neurons] += np.sum(
            self.weights[active_neurons] * incoming_spikes, axis=1
        )
        spiking_neurons = active_neurons & (self.membrane_potentials >= self.threshold)

        self.membrane_potentials[spiking_neurons] = 0.0
        self._spike_times[spiking_neurons & np.isinf(self._spike_times)] = current_time
        self.refractory_times[spiking_neurons] = self.refractory_period

        return spiking_neurons.astype(np.float32)

    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float):
        active_neurons = self._update_refractory(dt)
        return self._update_potential(incoming_spikes, current_time, active_neurons)

    def backward(self, pre_spike_times: np.ndarray):
        neurons_to_learn = (
            self.competition_mechanism.neurons_to_learn(self._spike_times)
            if self.competition_mechanism
            else range(self.num_outputs)
        )
        for neuron_idx in neurons_to_learn:
            self.weights[neuron_idx] = self.learning_mechanism.update_weights(
                self.weights[neuron_idx], pre_spike_times, self._spike_times[neuron_idx]
            )

        if self.threshold_adaptation:
            self.threshold = self.threshold_adaptation.update(
                self.threshold, self._spike_times
            )

    def reset(self):
        self.membrane_potentials.fill(0.0)
        self.refractory_times.fill(0.0)
        self._spike_times.fill(np.inf)

    @property
    def spike_times(self):
        return self._spike_times
