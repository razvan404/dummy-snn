import torch
import torch.nn.functional as F

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.learning.mechanism import LearningMechanism
from spiking.learning.competition import CompetitionMechanism
from spiking.threshold import ThresholdAdaptation


class ConvLearner:
    def __init__(
        self,
        layer: ConvIntegrateAndFireLayer,
        learning_mechanism: LearningMechanism | None = None,
        competition: CompetitionMechanism | None = None,
        threshold_adaptation: ThresholdAdaptation | None = None,
    ):
        self.layer = layer
        self.learning_mechanism = learning_mechanism
        self.competition = competition
        self.threshold_adaptation = threshold_adaptation

    def _per_filter_spike_times(self) -> torch.Tensor:
        """Reduce (F, oH, oW) spike times to per-filter earliest spike (F,)."""
        st = self.layer.spike_times  # (F, oH, oW)
        return st.flatten(1).min(dim=1).values  # (F,)

    def _select_neurons(self) -> torch.Tensor:
        per_filter = self._per_filter_spike_times()
        if self.competition:
            return self.competition.neurons_to_learn(per_filter)
        return torch.nonzero(torch.isfinite(per_filter), as_tuple=False).flatten()

    @torch.no_grad()
    def step(self, pre_spike_times: torch.Tensor) -> float:
        """Apply one learning step after a forward pass.

        Args:
            pre_spike_times: (C, H, W) input spike times.

        Returns:
            Average absolute weight change.
        """
        neurons_to_learn = self._select_neurons().flatten()
        self.neurons_to_learn = neurons_to_learn

        dw = 0.0
        if self.learning_mechanism and len(neurons_to_learn) > 0:
            dw = self._update_weights(neurons_to_learn, pre_spike_times)

        if self.threshold_adaptation and self.layer.training:
            per_filter = self._per_filter_spike_times()
            self.layer.thresholds.copy_(
                self.threshold_adaptation.update(
                    self.layer.thresholds,
                    per_filter,
                    neurons_to_learn=neurons_to_learn,
                    weights=self.layer.weights,
                    pre_spike_times=pre_spike_times,
                )
            )

        if not self.learning_mechanism:
            return 0.0
        return dw

    def _update_weights(
        self, neurons_to_learn: torch.Tensor, pre_spike_times: torch.Tensor
    ) -> float:
        """Apply STDP to winning filters, averaging across spatial positions.

        For each winning filter, unfold input times into patches and compute
        STDP updates per spatial position, then average.
        """
        C, H, W = pre_spike_times.shape
        kH = self.layer.kernel_size
        stride = self.layer.stride
        padding = self.layer.padding

        # Unfold input: (1, C, H, W) → (C*kH*kW, L) where L = oH*oW
        padded = F.pad(pre_spike_times.unsqueeze(0), [padding] * 4, value=float("inf"))
        patches = padded.unfold(2, kH, stride).unfold(
            3, kH, stride
        )  # (1, C, oH, oW, kH, kW)
        patches = patches.squeeze(0)  # (C, oH, oW, kH, kW)
        oH, oW = patches.shape[1], patches.shape[2]
        # Reshape to (oH*oW, C*kH*kW) — one row per spatial position
        patches = patches.permute(1, 2, 0, 3, 4).reshape(oH * oW, C * kH * kH)

        total_dw = 0.0
        n_updates = 0

        for f_idx in neurons_to_learn:
            f = f_idx.item()
            filter_weights = self.layer.weights[f]  # (C, kH, kW)
            filter_spike_times = self.layer.spike_times[f]  # (oH, oW)

            # Find positions where this filter spiked
            spiked_mask = torch.isfinite(filter_spike_times)
            if not spiked_mask.any():
                continue

            spiked_flat = spiked_mask.flatten()  # (L,)
            spiked_positions = torch.nonzero(spiked_flat, as_tuple=True)[0]

            # For each spiked position, compute weight update
            weight_updates = torch.zeros_like(filter_weights.flatten())
            for pos in spiked_positions:
                pre_times = patches[pos]  # (C*kH*kW,)
                post_time = filter_spike_times.flatten()[pos]

                updated = self.learning_mechanism.update_weights(
                    filter_weights.flatten().unsqueeze(0),
                    pre_times,
                    post_time.unsqueeze(0).unsqueeze(1),
                )
                weight_updates += updated.squeeze(0) - filter_weights.flatten()
                n_updates += 1

            # Average across positions
            avg_update = weight_updates / len(spiked_positions)
            weights_before = filter_weights.data.clone()
            new_weights = filter_weights.flatten() + avg_update

            if self.layer.training:
                self.layer.weights.data[f] = new_weights.reshape(filter_weights.shape)

            total_dw += torch.abs(avg_update).mean().item()

        if n_updates == 0:
            return 0.0
        return total_dw / len(neurons_to_learn)

    def learning_rate_step(self):
        if self.learning_mechanism:
            self.learning_mechanism.learning_rate_step()
        if self.threshold_adaptation:
            self.threshold_adaptation.learning_rate_step()
