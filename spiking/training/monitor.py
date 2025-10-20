import numpy as np
import torch
import matplotlib.pyplot as plt

from spiking import SpikingModule


class TrainingMonitor:
    def __init__(self, model: SpikingModule, splits: list[str] | None = None):
        if splits is None:
            splits = ["train", "val", "test"]
        self.model = model
        self.weight_diffs = {split: [] for split in splits}

        self.some_threshold_indices = [1, 10, 56, 99]
        self.thresholds = {
            "mean": [],
            "min": [],
            "max": [],
            **{f"idx_{idx}": [] for idx in self.some_threshold_indices},
        }

        self.neurons_activity = torch.zeros(self.model.num_outputs, dtype=torch.float32)

    def current_neurons_activity(self) -> torch.Tensor:
        spike_times = self.model.spike_times.cpu()
        finite_mask = torch.isfinite(spike_times)
        finite_spike_times = spike_times[finite_mask]

        if finite_spike_times.numel() == 0:
            return torch.tensor([], dtype=torch.long)

        topk_indices = torch.topk(
            -finite_spike_times, k=min(20, finite_spike_times.numel())
        ).indices
        finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()

        return finite_indices[topk_indices]

    def log(self, *, split: str, dw: float):
        self.weight_diffs[split].append(dw)

        if split != "train":
            return

        thresholds = self.model.thresholds
        self.thresholds["mean"].append(thresholds.mean().item())
        self.thresholds["min"].append(thresholds.min().item())
        self.thresholds["max"].append(thresholds.max().item())

        for idx in self.some_threshold_indices:
            if idx < len(thresholds):
                self.thresholds[f"idx_{idx}"].append(thresholds[idx].item())

        active = self.current_neurons_activity()
        self.neurons_activity[active] += 1.0

    def most_active_neurons(self, num_neurons: int = 20) -> torch.Tensor:
        return torch.topk(self.model.thresholds, num_neurons).indices

    def plot_weight_evolution(
        self, split: str, title: str = None, window_size: int = 100
    ):
        plt.scatter(
            np.arange(len(self.weight_diffs[split])),
            self.weight_diffs[split],
            s=1,
            label="Losses",
        )

        if len(self.weight_diffs[split]) >= window_size:
            moving_avg = np.convolve(
                self.weight_diffs[split],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            plt.plot(
                np.arange(window_size - 1, len(self.weight_diffs[split])),
                moving_avg,
                label=f"Moving Average (window={window_size})",
                linewidth=1,
                color="r",
            )

        if title:
            plt.title(title)
        plt.xlabel(f"{split} step")
        plt.ylabel("loss")
        plt.legend()

    def plot_thresholds_evolution(self, title: str = None):
        for metric, values in self.thresholds.items():
            style = "--" if metric.startswith("idx_") else "-"
            width = 1.5 if metric.startswith("idx_") else 2
            alpha = 0.9 if metric.startswith("idx_") else 1.0
            plt.plot(
                values, label=metric, linestyle=style, linewidth=width, alpha=alpha
            )

        if title:
            plt.title(title)
        plt.xlabel("Training Step")
        plt.ylabel("Threshold Value")
        plt.legend()

    def plot_neurons_activity(self):
        indices = torch.arange(self.model.num_outputs).cpu().numpy()
        activity = self.neurons_activity.cpu().numpy()

        plt.bar(indices, activity)
        plt.xlabel("Neuron Index")
        plt.ylabel("Activity")
        plt.title("Neuron Activity Bar Plot")

    def visualize_weights(
        self, image_shape: tuple[int, int], neurons_indices=None, ncols: int = 4
    ):
        if neurons_indices is None:
            neurons_indices = range(self.model.num_outputs)

        nrows = (len(neurons_indices) - 1) // ncols + 1
        plt.figure(figsize=(ncols * 2, nrows * 2))
        plt.suptitle("Weights")
        for idx, neuron_idx in enumerate(neurons_indices, start=1):
            img = self.model.weights[neuron_idx].reshape((2, *image_shape))
            padding = torch.zeros((1, *image_shape), device=img.device)
            img = torch.cat([img, padding], dim=0)
            img = img.permute(1, 2, 0).detach().cpu().numpy()

            plt.subplot(nrows, ncols, idx)
            plt.title(str(int(neuron_idx)))
            plt.axis("off")
            plt.imshow(img)
