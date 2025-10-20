from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SpikingModule(nn.Module, ABC):
    __slots__ = ["num_inputs", "num_outputs"]

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @property
    @abstractmethod
    def spike_times(self):
        pass

    @abstractmethod
    def forward(self, incoming_spikes: torch.Tensor, current_time: float, dt: float):
        pass

    @abstractmethod
    def backward(self, pre_spike_times: torch.Tensor):
        pass

    @abstractmethod
    def reset(self):
        pass
