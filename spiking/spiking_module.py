from abc import ABC, abstractmethod

import numpy as np


class SpikingModule(ABC):
    __slots__ = ["spike_times"]

    @abstractmethod
    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float): ...

    @abstractmethod
    def backward(self, pre_spike_times: np.ndarray): ...

    @abstractmethod
    def reset(self): ...
