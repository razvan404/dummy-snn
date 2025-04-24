import torch
import matplotlib.pyplot as plt

from ..layers import IntegrateAndFireOptimizedLayer


class Monitor:
    def __init__(self, model: IntegrateAndFireOptimizedLayer):
        self.model = model
        self.weight_diffs = []

        self.some_threshold_indices = [1, 10, 56, 99]
        self.thresholds = {
            "mean": [],
            "min": [],
            "max": [],
            **{f"idx_{idx}": [] for idx in self.some_threshold_indices},
        }

        self.neurons_activity = torch.zeros(
            self.model.num_outputs, dtype=torch.float32, device=self.model.device
        )

    def current_neurons_activity(self) -> torch.Tensor:
        spike_times = self.model.spike_times
        finite_mask = torch.isfinite(spike_times)
        finite_indices = torch.nonzero(finite_mask, as_tuple=False)

        sorted_finite = torch.argsort(spike_times[finite_mask])[:20]
        return finite_indices[sorted_finite]

    def log(self, *, loss: float) -> float:
        self.weight_diffs.append(loss)

        thresholds = self.model.thresholds
        self.thresholds["mean"].append(thresholds.mean().item())
        self.thresholds["min"].append(thresholds.min().item())
        self.thresholds["max"].append(thresholds.max().item())

        for idx in self.some_threshold_indices:
            if idx < len(thresholds):
                self.thresholds[f"idx_{idx}"].append(thresholds[idx].item())

        active = self.current_neurons_activity()
        self.neurons_activity[active] += 1.0

        return self.thresholds["mean"][-1]

    def most_active_neurons(self, num_neurons: int = 20) -> torch.Tensor:
        return torch.argsort(self.neurons_activity, descending=True)[:num_neurons]

    def plot_weight_evolution(self, title: str = None):
        plt.plot(self.weight_diffs)
        if title:
            plt.title(title)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.show()

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
        plt.show()

    def plot_neurons_activity(self):
        indices = torch.arange(self.model.num_outputs).cpu().numpy()
        activity = self.neurons_activity.cpu().numpy()

        plt.bar(indices, activity)
        plt.xlabel("Neuron Index")
        plt.ylabel("Activity")
        plt.title("Neuron Activity Bar Plot")
        plt.show()
