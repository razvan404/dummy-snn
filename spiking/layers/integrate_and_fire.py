import torch
import torch.nn as nn

from spiking.spiking_module import SpikingModule
from spiking.threshold import ThresholdInitialization

from .surrogate_spike import SurrogateSpike


class IntegrateAndFireLayer(SpikingModule):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        threshold_initialization: ThresholdInitialization,
        refractory_period: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs)

        self.refractory_period = refractory_period

        self.weights = nn.Parameter(torch.rand((num_outputs, num_inputs), dtype=dtype))
        self.thresholds = nn.Parameter(
            threshold_initialization.initialize((num_outputs,))
        )

        self.register_buffer(
            "membrane_potentials", torch.zeros((num_outputs,), dtype=dtype)
        )
        self.register_buffer(
            "refractory_times", torch.zeros((num_outputs,), dtype=dtype)
        )
        self.register_buffer(
            "_spike_times", torch.full((num_outputs,), float("inf"), dtype=dtype)
        )
        self.register_buffer("_output_spikes", torch.zeros((num_outputs,), dtype=dtype))

    def _update_refractory(self, dt: float) -> torch.Tensor:
        active_neurons = self.refractory_times == 0
        self.refractory_times.sub_(dt).clamp_(min=0.0)
        return active_neurons

    def _update_potential(
        self,
        incoming_spikes: torch.Tensor,
        current_time: float,
        active_neurons: torch.Tensor,
    ) -> torch.Tensor:
        self._output_spikes.zero_()

        if not active_neurons.any():
            return self._output_spikes

        spike_indices = incoming_spikes.nonzero(as_tuple=True)[0]
        if len(spike_indices) == 0:
            return self._output_spikes

        input_contrib = self.weights[active_neurons][:, spike_indices].sum(dim=1)
        self.membrane_potentials[active_neurons] += input_contrib

        potentials = self.membrane_potentials[active_neurons]
        thresholds = self.thresholds[active_neurons]

        if torch.is_grad_enabled():
            spikes_active = SurrogateSpike.apply(potentials, thresholds)
        else:
            spikes_active = (potentials >= thresholds).float()
        self._output_spikes[active_neurons] = spikes_active

        spiking_mask_full = self._output_spikes > 0.0
        self.membrane_potentials[spiking_mask_full] = 0.0

        unspiked_neurons = spiking_mask_full & torch.isinf(self._spike_times)
        self._spike_times[unspiked_neurons] = current_time
        self.refractory_times[spiking_mask_full] = self.refractory_period

        return self._output_spikes

    def forward(
        self, incoming_spikes: torch.Tensor, current_time: float, dt: float
    ) -> torch.Tensor:
        active_neurons = self._update_refractory(dt)
        return self._update_potential(incoming_spikes, current_time, active_neurons)

    def reset(self):
        self.membrane_potentials.zero_()
        self.refractory_times.zero_()
        self._spike_times.fill_(float("inf"))
        self._output_spikes.zero_()

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times
