from ..learning import LearningMechanism
from ..spiking_module import SpikingModule


class SpikingNeuron(SpikingModule):
    __slots__ = ["num_inputs", "weights", "learning_mechanism"]

    def __init__(self, num_inputs: int, learning_mechanism: LearningMechanism):
        self.num_inputs = num_inputs
        self.learning_mechanism = learning_mechanism
        self.weights = None
