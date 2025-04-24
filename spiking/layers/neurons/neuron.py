from abc import ABC

from spiking.learning import LearningMechanism
from spiking.spiking_module import SpikingModule


class SpikingNeuron(SpikingModule, ABC):
    __slots__ = ["learning_mechanism"]

    def __init__(self, num_inputs: int, learning_mechanism: LearningMechanism):
        super().__init__(num_inputs=num_inputs, num_outputs=1)
        self.learning_mechanism = learning_mechanism
