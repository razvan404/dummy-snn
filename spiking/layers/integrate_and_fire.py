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

    @torch.no_grad()
    def precompute_cumulative_potentials(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Precompute sorted unique times and cumulative potentials at group boundaries.

        Returns (unique_times, cum_at_boundaries) or None if no finite inputs.
        cum_at_boundaries shape: (num_outputs, num_groups).
        """
        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return None

        finite_indices = torch.nonzero(finite_mask, as_tuple=True)[0]
        finite_times = input_times[finite_indices]

        sorted_times, sort_order = finite_times.sort()
        sorted_indices = finite_indices[sort_order]

        sorted_weights = self.weights[:, sorted_indices]
        cum_potentials = sorted_weights.cumsum(dim=1)

        unique_times, counts = torch.unique_consecutive(
            sorted_times, return_counts=True
        )
        group_end_indices = counts.cumsum(dim=0) - 1
        cum_at_boundaries = cum_potentials[:, group_end_indices]

        return unique_times, cum_at_boundaries

    @staticmethod
    def spike_times_from_cumulative_potentials(
        unique_times: torch.Tensor,
        cum_at_boundaries: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve first spike times from precomputed cumulative potentials."""
        num_outputs = cum_at_boundaries.shape[0]
        result = torch.full((num_outputs,), float("inf"), dtype=unique_times.dtype)

        crossed = cum_at_boundaries >= thresholds.unsqueeze(1)
        any_crossed = crossed.any(dim=1)
        first_crossing = crossed.float().argmax(dim=1)
        result[any_crossed] = unique_times[first_crossing[any_crossed]]

        return result

    @torch.no_grad()
    def infer_spike_times(self, input_times: torch.Tensor) -> torch.Tensor:
        """Compute first spike times analytically without mutating model state."""
        precomputed = self.precompute_cumulative_potentials(input_times)
        if precomputed is None:
            return torch.full(
                (self.num_outputs,), float("inf"), dtype=input_times.dtype
            )
        return self.spike_times_from_cumulative_potentials(
            *precomputed, self.thresholds
        )

    @torch.no_grad()
    def infer_spike_times_and_potentials_batch(
        self, input_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analytical spike time inference, also returning final potentials.

        :param input_times: (B, num_inputs)
        :returns: (spike_times, cum_potential) both (B, num_outputs).
            cum_potential holds each neuron's cumulative membrane potential at the
            end of the input window (or at the moment it spiked).
        """
        B = input_times.shape[0]
        result = torch.full(
            (B, self.num_outputs), float("inf"), dtype=input_times.dtype
        )
        cum_potential = torch.zeros((B, self.num_outputs), dtype=input_times.dtype)

        finite_mask = torch.isfinite(input_times)
        if not finite_mask.any():
            return result, cum_potential

        unique_times = input_times[finite_mask].unique().sort()[0]

        not_yet_spiked = torch.ones((B, self.num_outputs), dtype=torch.bool)

        for t in unique_times:
            active = (input_times == t).float()
            contrib = active @ self.weights.T
            cum_potential += contrib

            crossed = (cum_potential >= self.thresholds) & not_yet_spiked
            result[crossed] = t
            not_yet_spiked &= ~crossed

            if not not_yet_spiked.any():
                break

        return result, cum_potential

    @torch.no_grad()
    def infer_spike_times_batch(self, input_times: torch.Tensor) -> torch.Tensor:
        """Batched analytical spike time inference.

        Iterates over unique input times and accumulates membrane potentials
        via batched matmul. Efficient when inputs are discretized to few bins.

        :param input_times: (B, num_inputs)
        :returns: (B, num_outputs)
        """
        result, _ = self.infer_spike_times_and_potentials_batch(input_times)
        return result

    @property
    def spike_times(self) -> torch.Tensor:
        return self._spike_times
