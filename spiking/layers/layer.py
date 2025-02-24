from abc import ABC

from ..competition import CompetitionMechanism
from ..learning import LearningMechanism
from ..spiking_module import SpikingModule


class SpikingLayer(SpikingModule, ABC):
    __slots__ = [
        "learning_mechanism",
        "competition_mechanism",
        "threshold_initialization",
        "threshold_adaptation",
    ]

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
    ):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs)
        self.learning_mechanism = learning_mechanism
        self.competition_mechanism = competition_mechanism
