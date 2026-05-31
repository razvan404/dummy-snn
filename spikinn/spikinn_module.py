from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SpikinnModule(nn.Module, ABC):
    _backend: str = "gather"

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @property
    @abstractmethod
    def spike_times(self):
        pass

    @abstractmethod
    def reset(self):
        pass
