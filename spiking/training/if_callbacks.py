import os

import torch
from matplotlib import pyplot as plt

from spiking.layers import IntegrateAndFireLayer
from spiking.training.callbacks_interface import CallbacksInterface
from spiking.training.monitor import TrainingMonitor
from spiking.visualization import SpikesVisualization


class IntegrateAndFireCallbacks(CallbacksInterface):
    def __init__(
        self,
        model: IntegrateAndFireLayer,
        visualize: bool = True,
        figures_dir: str | None = None,
    ):
        self.model = model
        self.epoch = None
        self.membrane_potentials = []
        self.membrane_potentials_times = []
        self.monitor = TrainingMonitor(model)
        self.visualize = visualize
        self.figures_dir = figures_dir
        if figures_dir:
            os.makedirs(figures_dir, exist_ok=True)

    def _should_visualize(self, batch_idx: int):
        return self.visualize and batch_idx < 4

    def callback_step_spike(
        self,
        batch_idx: int,
        current_time: float,
        output_spike: torch.Tensor,
        split: str,
    ) -> bool:
        if not self._should_visualize(batch_idx):
            return False

        if torch.any(output_spike == 1.0):
            self.membrane_potentials.append(
                [
                    (
                        self.model.thresholds[idx]
                        if output_spike[idx]
                        else self.model.membrane_potentials[idx]
                    )
                    for idx in range(self.model.num_outputs)
                ]
            )
            self.membrane_potentials_times.append(current_time)
        self.membrane_potentials.append(self.model.membrane_potentials.clone())
        self.membrane_potentials_times.append(current_time)

        return True

    def callback_step(
        self,
        batch_idx: int,
        pre_spike_times: torch.Tensor,
        dw: float,
        label: str | None,
        split: str,
    ):
        self.monitor.log(split=split, dw=dw)

        if not self._should_visualize(batch_idx):
            return

        post_spike_times = self.model.spike_times

        if batch_idx == 0:
            plt.figure(figsize=(20, 10))
            plt.suptitle(f"Epoch {self.epoch}. Split {split}")

        plt.subplot(2, 4, 1 + 2 * batch_idx)
        SpikesVisualization.plot_pre_post_spikes(
            pre_spike_times.numpy(),
            post_spike_times.numpy(),
            title=f"Pre- vs Post-synaptic. digit {label}",
            unique_colors=True,
        )

        plt.subplot(2, 4, 2 + 2 * batch_idx)
        SpikesVisualization.plot_multiple_membrane_potentials(
            self.membrane_potentials,
            self.membrane_potentials_times,
            title=f"Membrane Potential Over Time. digit {label}",
        )

        if batch_idx == 3:
            plt.tight_layout()
            if self.figures_dir is None:
                plt.show()
            else:
                plt.savefig(f"{self.figures_dir}/{split}-e{self.epoch}.png")
                plt.close()

        self.membrane_potentials = []
        self.membrane_potentials_times = []
