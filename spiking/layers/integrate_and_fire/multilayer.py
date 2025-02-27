from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.threshold import (
    ThresholdInitialization,
    ThresholdAdaptation,
    ConstantInitialization,
)

from ..sequential import SpikingSequential
from .optimized_layer import IntegrateAndFireOptimizedLayer


class IntegrateAndFireMultilayer(SpikingSequential):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: list[int],
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold: float | list[float] = 1.0,
        refractory_period: float = 1.0,
        threshold_initialization: ThresholdInitialization | None = None,
        threshold_adaptation: ThresholdAdaptation | None = ConstantInitialization(),
    ):
        if isinstance(threshold, list):
            assert len(threshold) == len(num_hidden) + 1
        else:
            threshold = [threshold for _ in range(len(num_hidden) + 1)]

        layers = [
            IntegrateAndFireOptimizedLayer(
                num_inputs=num_inputs,
                num_outputs=num_outputs if len(num_hidden) == 0 else num_hidden[0],
                learning_mechanism=learning_mechanism,
                threshold=threshold[0],
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
                IntegrateAndFireOptimizedLayer(
                    num_inputs=layer_inputs,
                    num_outputs=layer_outputs,
                    learning_mechanism=learning_mechanism,
                    competition_mechanism=competition_mechanism,
                    threshold=threshold[i + 1],
                    refractory_period=refractory_period,
                )
            )
        super().__init__(*layers)
