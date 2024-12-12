from abc import ABC, abstractmethod

import numpy as np

from ..competition import CompetitionMechanism
from ..learning import LearningMechanism


class SpikingLayer(ABC):
    def __init__(self, num_inputs: int = 1, num_outputs: int = 1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @abstractmethod
    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float): ...

    @abstractmethod
    def backward(
        self,
        pre_spike_times: np.ndarray,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
    ): ...
