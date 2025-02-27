from abc import ABC

from ..competition import CompetitionMechanism
from ..learning import LearningMechanism
from ..spiking_module import SpikingModule
from ..threshold import ThresholdInitialization, ThresholdAdaptation


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
        threshold_initialization: ThresholdInitialization | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
    ):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs)
        self.learning_mechanism = learning_mechanism
        self.competition_mechanism = competition_mechanism
        self.threshold_initialization = threshold_initialization
        self.threshold_adaptation = threshold_adaptation
