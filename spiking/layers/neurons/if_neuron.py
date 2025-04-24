import torch

from spiking.learning import LearningMechanism
from .neuron import SpikingNeuron


class IntegrateAndFireNeuron(SpikingNeuron):
    def __init__(
        self,
        num_inputs: int,
        learning_mechanism: LearningMechanism,
        threshold: float = 1.0,
        refractory_period: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_inputs=num_inputs, learning_mechanism=learning_mechanism)

        self.threshold = torch.tensor(threshold, device=self.device)
        self.refractory_period = refractory_period
        self.device = device

        self.membrane_potential = torch.tensor(0.0, device=self.device)
        self.refractory_time = torch.tensor(0.0, device=self.device)
        self.weights = torch.abs(torch.rand(num_inputs, device=self.device))

        self._spike_times = []

    def _update_refractory(self, dt: float) -> bool:
        if self.refractory_time > 0:
            self.refractory_time -= dt
            return False
        return True

    def _update_potential(
        self, incoming_spikes: torch.Tensor, current_time: float
    ) -> torch.Tensor:
        self.membrane_potential += torch.sum(self.weights[incoming_spikes == 1.0])
        if self.membrane_potential < self.threshold:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.membrane_potential = torch.tensor(0.0, device=self.device)
        self.refractory_time = torch.tensor(self.refractory_period, device=self.device)
        self._spike_times.append(current_time)

        return torch.tensor(1.0, dtype=torch.float32, device=self.device)

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        if not self._update_refractory(dt):
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        return self._update_potential(incoming_spikes, current_time)

    def backward(self, pre_spike_times: torch.Tensor):
        for spike_time in self._spike_times:
            self.weights = self.learning_mechanism.update_weights(
                self.weights, pre_spike_times, spike_time
            )

    def reset(self):
        self.membrane_potential = torch.tensor(0.0, device=self.device)
        self.refractory_time = torch.tensor(0.0, device=self.device)
        self._spike_times = []

    @property
    def spike_times(self):
        return self._spike_times
