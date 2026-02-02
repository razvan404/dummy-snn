import torch
from torch.utils.data import DataLoader

from spiking import iterate_spikes


class STDPThresholdOptimizer:
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_threshold: float = 1.0,
        max_threshold: float = 100.0,
    ):
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def compute_balance(
        self,
        weights: torch.Tensor,
        pre_spike_times: torch.Tensor,
        post_spike_time: float,
    ) -> int:
        """
        Compute potentiation - depression count for a single neuron.
        Only considers synapses with weights above the midpoint.

        # TODO: instead of sum, weight contribution of input spike based on weight.
        # Experiment!!
        """
        weight_threshold = 0.5
        significant_mask = (
            weights > weight_threshold
        )  # hardware wise: harder to implement.

        delta_t = post_spike_time - pre_spike_times
        delta_t_significant = delta_t[significant_mask]

        n_pot = (delta_t_significant > 0).sum().item()
        n_dep = (delta_t_significant < 0).sum().item()
        return n_pot - n_dep

    def step(
        self,
        model,
        pre_spike_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update thresholds for neurons that spiked based on STDP balance.
        """
        threshold_deltas = torch.zeros_like(model.thresholds)

        spiked_mask = torch.isfinite(model.spike_times)
        spiked_indices = torch.nonzero(spiked_mask, as_tuple=True)[0]

        for neuron_idx in spiked_indices:
            idx = neuron_idx.item()
            post_spike_time = model.spike_times[idx].item()
            neuron_weights = model.weights[idx]

            balance = self.compute_balance(
                neuron_weights, pre_spike_times, post_spike_time
            )

            if balance != 0:
                delta = -self.learning_rate * (1 if balance > 0 else -1)
                threshold_deltas[idx] = delta

        new_thresholds = model.thresholds + threshold_deltas
        new_thresholds = torch.clamp(
            new_thresholds, self.min_threshold, self.max_threshold
        )
        model.thresholds.copy_(new_thresholds)

        return threshold_deltas

    def optimize(
        self,
        model,
        data_loader: DataLoader,
        image_shape: tuple,
        num_epochs: int = 10,
        convergence_threshold: float = 0.001,
        verbose: bool = True,
        debug: bool = False,
    ) -> dict:
        """
        Run optimization until convergence or max epochs.
        """
        device = next(model.parameters()).device
        model.eval()
        model.updatable_weights = False
        model.updatable_thresholds = False

        history = {
            "mean_delta": [],
            "thresholds": [model.thresholds.detach().cpu().clone()],
        }

        for epoch in range(num_epochs):
            epoch_deltas = []
            total_neurons_spiked = 0
            total_samples = 0

            for sample_idx, (spikes, label, times) in enumerate(data_loader):
                model.reset()

                for incoming_spikes, current_time, dt in iterate_spikes(
                    spikes, shape=image_shape
                ):
                    model.forward(
                        incoming_spikes.flatten().to(device), current_time, dt
                    )

                num_spiked = torch.isfinite(model.spike_times).sum().item()
                total_neurons_spiked += num_spiked
                total_samples += 1

                if debug and sample_idx < 5:
                    print(f"  Sample {sample_idx}: {num_spiked} neurons spiked")

                pre_spike_times = times.flatten().to(device)
                deltas = self.step(model, pre_spike_times)
                epoch_deltas.append(deltas.abs().mean().item())

            mean_delta = sum(epoch_deltas) / len(epoch_deltas) if epoch_deltas else 0
            avg_spiked = (
                total_neurons_spiked / total_samples if total_samples > 0 else 0
            )
            history["mean_delta"].append(mean_delta)
            history["thresholds"].append(model.thresholds.detach().cpu().clone())

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: mean |delta| = {mean_delta:.6f}, avg neurons spiked = {avg_spiked:.1f}"
                )

            if mean_delta < convergence_threshold:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

        return history
