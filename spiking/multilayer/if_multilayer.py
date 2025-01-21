import numpy as np

from .multilayer import SpikingMultilayer
from ..competition import CompetitionMechanism
from ..layers import IntegrateAndFireLayer
from ..learning import LearningMechanism


class IntegrateAndFireMultilayer(SpikingMultilayer):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: list[int],
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
    ):
        super().__init__(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            learning_mechanism=learning_mechanism,
            competition_mechanism=competition_mechanism,
        )
        self.layers = [
            IntegrateAndFireLayer(
                num_inputs=num_inputs,
                num_outputs=num_outputs if len(num_hidden) == 0 else num_hidden[0],
                learning_mechanism=learning_mechanism,
                threshold=threshold,
                refractory_period=refractory_period,
            )
        ]
        for i in range(len(num_hidden)):
            layer_inputs = num_hidden[i]
            layer_outputs = (
                num_hidden[i + 1] if i < len(num_hidden) - 1 else num_outputs
            )
            self.layers.append(
                IntegrateAndFireLayer(
                    num_inputs=layer_inputs,
                    num_outputs=layer_outputs,
                    learning_mechanism=learning_mechanism,
                    threshold=threshold,
                    refractory_period=refractory_period,
                )
            )

    @property
    def spike_times(self):
        return self.layers[-1].spike_times

    def forward(self, incoming_spikes: np.ndarray, current_time: float, dt: float):
        for layer in self.layers:
            incoming_spikes = layer.forward(incoming_spikes, current_time, dt)
        return incoming_spikes

    def backward(self, pre_spike_times: np.ndarray):
        for layer in self.layers:
            layer.backward(pre_spike_times)
            pre_spike_times = layer.spike_times

    def reset(self):
        for layer in self.layers:
            layer.reset()
