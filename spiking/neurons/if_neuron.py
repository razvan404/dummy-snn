import numpy as np

from ..learning import LearningMechanism
from ..neurons.neuron import SpikingNeuron


class IntegrateAndFireNeuron(SpikingNeuron):
    def __init__(
        self,
        num_inputs: int,
        learning_mechanism: LearningMechanism,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
    ):
        """
        Integrate and Fire Neuron Model.

        Args:
            threshold (float): Voltage threshold for firing.
            refractory_period (float): Amount of time it takes to recharge the neuron.
            num_inputs (int): The number of inputs of the neuron.
        """
        super().__init__(num_inputs=num_inputs, learning_mechanism=learning_mechanism)

        self.threshold = threshold
        self.membrane_potential = 0.0

        self.refractory_period = refractory_period
        self.refractory_time = 0.0
        self.weights = np.abs(np.random.rand(self.num_inputs))

        self._spike_times = []

    def _update_refractory(self, dt: float):
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False
        return True

    def _update_potential(self, incoming_spikes: np.ndarray, current_time: float):
        self.membrane_potential += np.sum(self.weights[incoming_spikes == 1.0])
        if self.membrane_potential < self.threshold:
            return np.float32(0.0)

        self.membrane_potential = 0.0
        self.refractory_time = self.refractory_period
        self._spike_times.append(current_time)

        return np.float32(1.0)

    def forward(
        self, incoming_spikes: np.ndarray, current_time: float, dt: float
    ) -> np.floating:
        if not self._update_refractory(dt):
            return np.float32(0.0)

        return self._update_potential(incoming_spikes, current_time)

    def backward(self, pre_spike_times: np.ndarray):
        for spike_time in self._spike_times:
            self.weights = self.learning_mechanism.update_weights(
                self.weights, pre_spike_times, spike_time
            )

    def reset(self):
        self.membrane_potential = 0.0
        self.refractory_time = 0.0
        self._spike_times = []

    @property
    def spike_times(self):
        return self._spike_times
