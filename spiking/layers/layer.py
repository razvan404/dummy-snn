from abc import ABC

from ..competition import CompetitionMechanism
from ..learning import LearningMechanism
from ..spiking_module import SpikingModule
from ..threshold import ThresholdInitialization, ThresholdAdaptation


class SpikingLayer(SpikingModule, ABC):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        threshold_initialization: ThresholdInitialization,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
    ):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs)
        self.learning_mechanism = learning_mechanism
        self.threshold_initialization = threshold_initialization
        self.competition_mechanism = competition_mechanism
        self.threshold_adaptation = threshold_adaptation
