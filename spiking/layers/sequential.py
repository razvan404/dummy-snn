import numpy as np

from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.spiking_module import SpikingModule


class SpikingSequential(SpikingModule):
    __slots__ = ["layers"]

    def __init__(self, *layers: SpikingModule):
        self.layers = list(layers)
        assert len(self.layers) >= 1
        super().__init__(
            num_inputs=self.layers[0].num_inputs,
            num_outputs=self.layers[-1].num_outputs,
        )

    @property
    def spike_times(self):
        return self.layers[-1].spike_times

    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float):
        for layer in self.layers:
            incoming_spikes = layer.forward(incoming_spikes, current_time, dt)
        return incoming_spikes

    def backward(self, pre_spike_times: np.ndarray):
        for layer in self.layers:
            layer.backward(pre_spike_times)
            pre_spike_times = layer.spike_times

    def reset(self):
        for layer in self.layers:
            layer.reset()
