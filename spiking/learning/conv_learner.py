import torch

from .base import BaseLearner
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer


class ConvLearner(BaseLearner):
    layer: ConvIntegrateAndFireLayer

    def _get_spike_times(self) -> torch.Tensor:
        st = self.layer.spike_times
        return st.flatten(1).min(dim=1).values

    def _update_weights(
        self, neurons_to_learn: torch.Tensor, pre_spike_times: torch.Tensor
    ) -> torch.Tensor:
        patches = self.layer._unfold_patches(pre_spike_times)
        L = patches.shape[0]
        dim = patches.shape[1]
        device = patches.device
        zero = torch.zeros((), device=device, dtype=patches.dtype)

        n_win = len(neurons_to_learn)
        if n_win == 0:
            return zero

        win_spike_times = self.layer.spike_times[neurons_to_learn].flatten(1)
        win_weights = self.layer.weights[neurons_to_learn]

        if n_win == 1 and L == 1:
            if win_spike_times.dim() >= 2:
                has_spike = torch.isfinite(win_spike_times[0, 0])
            else:
                has_spike = torch.isfinite(win_spike_times[0])
            if not has_spike:
                return zero
            updated = self.learning_mechanism.update_weights(
                win_weights,
                patches,
                win_spike_times,
            )
            deltas = updated - win_weights
            if self.layer.training:
                self.layer.weights.data[neurons_to_learn] = updated
            return torch.abs(deltas).mean()

        spiked_mask = torch.isfinite(win_spike_times)
        n_spiked = spiked_mask.sum(dim=1)
        has_spikes = n_spiked > 0
        if not has_spikes.any():
            return zero

        pre_times = patches.unsqueeze(0).expand(n_win, -1, -1)
        post_times = win_spike_times.unsqueeze(2)
        w_expanded = win_weights.unsqueeze(1).expand(-1, L, -1)

        updated = self.learning_mechanism.update_weights(
            w_expanded.reshape(n_win * L, dim),
            pre_times.reshape(n_win * L, dim),
            post_times.reshape(n_win * L, 1),
        )
        deltas = (updated - w_expanded.reshape(n_win * L, dim)).reshape(n_win, L, dim)

        masked_deltas = deltas * spiked_mask.unsqueeze(2)
        avg_deltas = masked_deltas.sum(dim=1) / n_spiked.clamp(min=1).unsqueeze(1)

        if self.layer.training:
            active = neurons_to_learn[has_spikes]
            new_weights = win_weights[has_spikes] + avg_deltas[has_spikes]
            self.layer.weights.data[active] = new_weights

        active_deltas = avg_deltas[has_spikes]
        return torch.abs(active_deltas).mean()
