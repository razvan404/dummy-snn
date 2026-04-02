import torch

from .base import BaseLearner
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer


class ConvLearner(BaseLearner):
    """Learner for convolutional spiking layers.

    Reduces per-position spike times to per-filter and applies STDP
    with spatial averaging across spiked positions.
    """

    layer: ConvIntegrateAndFireLayer

    def _get_spike_times(self) -> torch.Tensor:
        """Reduce (F, oH, oW) spike times to per-filter earliest spike (F,)."""
        st = self.layer.spike_times  # (F, oH, oW)
        return st.flatten(1).min(dim=1).values  # (F,)

    def _update_weights(
        self, neurons_to_learn: torch.Tensor, pre_spike_times: torch.Tensor
    ) -> float:
        """Apply STDP to winning filters, averaging across spatial positions.

        Processes all winning filters and their spiked spatial positions in a
        single batched STDP call, then averages weight deltas per filter.
        """
        # Unfold input into patches via layer helper: (L, dim)
        patches = self.layer._unfold_patches(pre_spike_times)  # (L, dim)
        L = patches.shape[0]
        dim = patches.shape[1]

        n_win = len(neurons_to_learn)
        if n_win == 0:
            return 0.0

        # Gather spike times and weights for all winning filters
        win_spike_times = self.layer.spike_times[neurons_to_learn].flatten(
            1
        )  # (n_win, L)
        win_weights = self.layer.weights[neurons_to_learn]  # (n_win, dim) — already 2D

        spiked_mask = torch.isfinite(win_spike_times)  # (n_win, L)
        n_spiked = spiked_mask.sum(dim=1)  # (n_win,)
        has_spikes = n_spiked > 0
        if not has_spikes.any():
            return 0.0

        # Expand patches across winning filters: (n_win, L, dim)
        pre_times = patches.unsqueeze(0).expand(n_win, -1, -1)
        post_times = win_spike_times.unsqueeze(2)  # (n_win, L, 1)
        w_expanded = win_weights.unsqueeze(1).expand(-1, L, -1)  # (n_win, L, dim)

        # Single batched STDP call over all (filter, position) pairs
        updated = self.learning_mechanism.update_weights(
            w_expanded.reshape(n_win * L, dim),
            pre_times.reshape(n_win * L, dim),
            post_times.reshape(n_win * L, 1),
        )
        deltas = (updated - w_expanded.reshape(n_win * L, dim)).reshape(n_win, L, dim)

        # Mask non-spiked positions and average per filter
        masked_deltas = deltas * spiked_mask.unsqueeze(2)
        avg_deltas = masked_deltas.sum(dim=1) / n_spiked.clamp(min=1).unsqueeze(1)

        # Update weights
        if self.layer.training:
            active = neurons_to_learn[has_spikes]
            new_weights = win_weights[has_spikes] + avg_deltas[has_spikes]
            self.layer.weights.data[active] = new_weights

        active_deltas = avg_deltas[has_spikes]
        return torch.abs(active_deltas).mean().item()
