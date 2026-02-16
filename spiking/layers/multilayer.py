from spiking.threshold import ThresholdInitialization

from .sequential import SpikingSequential
from .integrate_and_fire import IntegrateAndFireLayer


class IntegrateAndFireMultilayer(SpikingSequential):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: list[int],
        num_outputs: int,
        threshold_initialization: ThresholdInitialization,
        refractory_period: float = 1.0,
    ):
        all_sizes = [num_inputs] + num_hidden + [num_outputs]
        layers = [
            IntegrateAndFireLayer(
                num_inputs=all_sizes[i],
                num_outputs=all_sizes[i + 1],
                threshold_initialization=threshold_initialization,
                refractory_period=refractory_period,
            )
            for i in range(len(all_sizes) - 1)
        ]
        super().__init__(*layers)
