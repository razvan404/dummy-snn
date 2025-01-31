import numpy as np

from .layer import SpikingLayer
from ..competition import CompetitionMechanism
from ..learning import LearningMechanism
from ..neurons import IntegrateAndFireNeuron


class IntegrateAndFireLayer(SpikingLayer):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_mechanism=learning_mechanism,
            competition_mechanism=competition_mechanism,
        )
        self.neurons = [
            IntegrateAndFireNeuron(
                num_inputs=num_inputs,
                learning_mechanism=learning_mechanism,
                threshold=threshold,
                refractory_period=refractory_period,
            )
            for _ in range(num_outputs)
        ]
        self.spike_times = np.ones(self.num_outputs, dtype=np.float32) * np.inf

    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float):
        spikes = np.zeros(self.num_outputs, dtype=np.float32)

        for neuron_idx, neuron in enumerate(self.neurons):
            neuron_spike = neuron.forward(
                incoming_spikes=incoming_spikes, current_time=current_time, dt=dt
            )
            if neuron_spike == 1.0 and np.isinf(self.spike_times[neuron_idx]):
                self.spike_times[neuron_idx] = current_time
                spikes[neuron_idx] = 1.0

        return spikes

    def backward(self, pre_spike_times: np.ndarray):
        neurons_to_learn = (
            self.competition_mechanism.neurons_to_learn(self.spike_times)
            if self.competition_mechanism
            else range(0, self.num_outputs)
        )
        for neuron_idx in neurons_to_learn:
            self.neurons[neuron_idx].backward(pre_spike_times)

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()
        self.spike_times = np.ones(self.num_outputs, dtype=np.float32) * np.inf
