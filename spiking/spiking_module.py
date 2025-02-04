from abc import ABC, abstractmethod

import numpy as np


class SpikingModule(ABC):
    __slots__ = ["num_inputs", "num_outputs"]

    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @property
    @abstractmethod
    def spike_times(self): ...

    @abstractmethod
    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float): ...

    @abstractmethod
    def backward(self, pre_spike_times: np.ndarray): ...

    @abstractmethod
    def reset(self): ...
