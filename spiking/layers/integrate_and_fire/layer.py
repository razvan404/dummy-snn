import torch

from spiking.layers.layer import SpikingLayer
from spiking.competition import CompetitionMechanism
from spiking.learning import LearningMechanism
from spiking.threshold import (
    ThresholdInitialization,
    ThresholdAdaptation,
    ConstantInitialization,
)

from ..neurons import IntegrateAndFireNeuron


class IntegrateAndFireLayer(SpikingLayer):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        learning_mechanism: LearningMechanism,
        competition_mechanism: CompetitionMechanism | None = None,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
        threshold_initialization: ThresholdInitialization | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        if threshold_initialization is None:
            threshold_initialization = ConstantInitialization()

        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            learning_mechanism=learning_mechanism,
            competition_mechanism=competition_mechanism,
            threshold_initialization=threshold_initialization,
            threshold_adaptation=threshold_adaptation,
        )

        self.device = device
        self.neurons = [
            IntegrateAndFireNeuron(
                num_inputs=num_inputs,
                learning_mechanism=learning_mechanism,
                threshold=threshold_initialization.initialize(threshold),
                refractory_period=refractory_period,
                device=self.device,
            )
            for _ in range(num_outputs)
        ]

        self._spike_times = torch.full(
            (self.num_outputs,), float("inf"), dtype=torch.float32, device=self.device
        )

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        spikes = torch.zeros(self.num_outputs, dtype=torch.float32, device=self.device)

        for neuron_idx, neuron in enumerate(self.neurons):
            spike = neuron.forward(incoming_spikes, current_time, dt)
            if spike == 1.0 and torch.isinf(self._spike_times[neuron_idx]):
                self._spike_times[neuron_idx] = current_time
                spikes[neuron_idx] = 1.0

        return spikes

    def backward(self, pre_spike_times: torch.Tensor):
        neurons_to_learn = (
            self.competition_mechanism.neurons_to_learn(self._spike_times)
            if self.competition_mechanism
            else range(0, self.num_outputs)
        )

        for neuron_idx in neurons_to_learn:
            self.neurons[neuron_idx].backward(pre_spike_times)

        if self.threshold_adaptation:
            thresholds = torch.tensor(
                [neuron.threshold for neuron in self.neurons],
                dtype=torch.float32,
                device=self.device,
            )
            updated_thresholds = self.threshold_adaptation.update(
                thresholds,
                self._spike_times,
                neurons_to_learn=neurons_to_learn,
            )
            for neuron_idx, neuron in enumerate(self.neurons):
                neuron.threshold = updated_thresholds[neuron_idx]

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()
        self._spike_times.fill_(float("inf"))

    @property
    def spike_times(self):
        return self._spike_times
