from abc import ABC, abstractmethod

import numpy as np

from spiking.learning import LearningMechanism


class SpikingNeuron(ABC):
    __slots__ = ["num_inputs", "weights"]

    def __init__(self, num_inputs: int = 1):
        self.num_inputs = num_inputs
        self.weights = None

    @abstractmethod
    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float): ...

    @abstractmethod
    def backward(
        self, pre_spike_times: np.ndarray, learning_mechanism: LearningMechanism
    ): ...

    @abstractmethod
    def reset(self): ...
