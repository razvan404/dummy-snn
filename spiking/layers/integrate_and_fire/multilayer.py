from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.threshold import (
    ThresholdInitialization,
    ThresholdAdaptation,
)

from ..sequential import SpikingSequential
from .layer import IntegrateAndFireLayer


class IntegrateAndFireMultilayer(SpikingSequential):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: list[int],
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        refractory_period: float = 1.0,
        threshold_initialization: ThresholdInitialization | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
    ):
        layers = [
            IntegrateAndFireLayer(
                num_inputs=num_inputs,
                num_outputs=num_outputs if len(num_hidden) == 0 else num_hidden[0],
                learning_mechanism=learning_mechanism,
                refractory_period=refractory_period,
                threshold_initialization=threshold_initialization,
                threshold_adaptation=threshold_adaptation,
            )
        ]
        for i in range(len(num_hidden)):
            layer_inputs = num_hidden[i]
            layer_outputs = (
                num_hidden[i + 1] if i < len(num_hidden) - 1 else num_outputs
            )
            layers.append(
                IntegrateAndFireLayer(
                    num_inputs=layer_inputs,
                    num_outputs=layer_outputs,
                    learning_mechanism=learning_mechanism,
                    competition_mechanism=competition_mechanism,
                    refractory_period=refractory_period,
                )
            )
        super().__init__(*layers)
