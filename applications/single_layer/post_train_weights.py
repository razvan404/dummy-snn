import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset
from applications.single_layer.visualize import plot_comparison
from spiking import iterate_spikes
from spiking.learning import Learner, STDP
from spiking.utils import load_model, save_model


def plot_weights_evolution(
    weights_history: list[torch.Tensor],
    output_dir: str,
):
    """Plot weight evolution over epochs."""
    weights_array = np.array([w.numpy().flatten() for w in weights_history])
    num_epochs = weights_array.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Weight change per epoch
    ax = axes[0, 0]
    if num_epochs > 1:
        weight_changes = np.abs(np.diff(weights_array, axis=0)).mean(axis=1)
        ax.plot(range(1, len(weight_changes) + 1), weight_changes, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean |Weight Change|")
        ax.set_title("Weight Change per Epoch")
        ax.grid(True, alpha=0.3)

    # Mean weight evolution
    ax = axes[0, 1]
    mean_w = weights_array.mean(axis=1)
    std_w = weights_array.std(axis=1)
    epochs = np.arange(num_epochs)
    ax.plot(epochs, mean_w, label="Mean", color="blue")
    ax.fill_between(epochs, mean_w - std_w, mean_w + std_w, alpha=0.3, color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title("Mean Weight ± Std")
    ax.legend()

    # Weight distribution: start vs end
    ax = axes[1, 0]
    ax.hist(weights_array[0], bins=50, alpha=0.5, label="Start", color="blue")
    ax.hist(weights_array[-1], bins=50, alpha=0.5, label="End", color="orange")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight Distribution: Start vs End")
    ax.legend()

    # Weight change heatmap (sample of weights)
    ax = axes[1, 1]
    n_weights_to_show = min(100, weights_array.shape[1])
    sample_indices = np.linspace(
        0, weights_array.shape[1] - 1, n_weights_to_show, dtype=int
    )
    sampled_weights = weights_array[:, sample_indices].T
    im = ax.imshow(sampled_weights, aspect="auto", cmap="viridis")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Index (sampled)")
    ax.set_title("Weight Evolution Heatmap")
    plt.colorbar(im, ax=ax, label="Weight")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weights_evolution.png"), dpi=150)
    plt.close()


def plot_convergence(mean_dw: list[float], output_dir: str):
    """Plot convergence of weight updates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mean_dw) + 1), mean_dw, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean |Weight Delta|")
    ax.set_title("Convergence of Weight Updates")
    if all(d > 0 for d in mean_dw):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
    plt.close()


def plot_neuron_weights(
    weights: torch.Tensor,
    spike_counts: torch.Tensor,
    output_dir: str,
    image_shape: tuple[int, int],
    n_neurons: int = 16,
):
    """Plot weight receptive fields for the most firing neurons."""
    num_inputs = weights.shape[1]
    n_channels = num_inputs // (image_shape[0] * image_shape[1])

    # Get indices of most firing neurons
    top_indices = torch.argsort(spike_counts, descending=True)[:n_neurons]

    # Determine grid size
    n_cols = 4
    n_rows = (n_neurons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, neuron_idx in enumerate(top_indices):
        ax = axes[i]
        neuron_weights = weights[neuron_idx].detach().cpu().numpy()

        # Reshape to (channels, H, W) and take difference of on/off channels
        reshaped = neuron_weights.reshape(n_channels, *image_shape)
        if n_channels == 2:
            # on - off channels
            rf = reshaped[0] - reshaped[1]
        else:
            rf = reshaped[0]

        vmax = np.abs(rf).max()
        ax.imshow(rf, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"N{neuron_idx.item()} ({spike_counts[neuron_idx].item()} spikes)")
        ax.axis("off")

    # Hide unused axes
    for i in range(n_neurons, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Receptive Fields (Most Firing Neurons)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "neuron_weights.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Post-train weights using STDP on all spiking neurons"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the pre-trained model (.pth file)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of post-training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="STDP learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=1.0,
        help="Learning rate decay per epoch (default: 1.0, no decay)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_model(args.model_path)
    print(f"Loaded model from {args.model_path}")
    num_neurons = model.weights.shape[0]
    print(f"Model has {num_neurons} neurons")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.model_path), "stdp_w_post"
    )
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader = create_dataset("mnist_subset")
    image_shape = train_loader.dataset.image_shape
    spike_shape = (2, *image_shape)

    # Record thresholds before to verify they don't change
    thresholds_before = model.thresholds.detach().cpu().clone()

    print("\nEvaluating model BEFORE post-training...")
    before_train, before_val = evaluate_model(
        copy.deepcopy(model), train_loader, val_loader, image_shape=spike_shape
    )
    print(f"  Train accuracy: {before_train['accuracy']:.4f}")
    print(f"  Val accuracy: {before_val['accuracy']:.4f}")

    device = torch.device("cpu")
    model = model.to(device)

    stdp = STDP(
        tau_pre=0.1,
        tau_post=0.1,
        max_pre_spike_time=1.0,
        learning_rate=args.learning_rate,
        decay_factor=args.decay_factor,
    )
    learner = Learner(model, stdp)

    print(f"\nRunning STDP weight optimization...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Decay factor: {args.decay_factor}")
    print(f"  Epochs: {args.num_epochs}")

    num_neurons = model.weights.shape[0]
    spike_counts = torch.zeros(num_neurons, dtype=torch.long)

    history = {
        "mean_dw": [],
        "weights": [model.weights.detach().cpu().clone()],
    }

    model.train()
    for epoch in range(args.num_epochs):
        epoch_dw = []
        total_neurons_spiked = 0
        total_samples = 0

        for times, label in train_loader:
            model.reset()

            for incoming_spikes, current_time, dt in iterate_spikes(times):
                model.forward(incoming_spikes.flatten().to(device), current_time, dt)

            spiked_mask = torch.isfinite(model.spike_times)
            spike_counts += spiked_mask.cpu().long()
            num_spiked = spiked_mask.sum().item()
            total_neurons_spiked += num_spiked
            total_samples += 1

            pre_spike_times = times.flatten().to(device)
            dw = learner.step(pre_spike_times)
            epoch_dw.append(dw)

            model.reset()

        mean_dw = sum(epoch_dw) / len(epoch_dw) if epoch_dw else 0
        avg_spiked = total_neurons_spiked / total_samples if total_samples > 0 else 0

        history["mean_dw"].append(mean_dw)
        history["weights"].append(model.weights.detach().cpu().clone())

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"mean |dw| = {mean_dw:.6f}, "
            f"avg neurons spiked = {avg_spiked:.1f}, "
            f"lr = {stdp.learning_rate:.6f}"
        )

        learner.learning_rate_step()

    history["spike_counts"] = spike_counts

    # Verify thresholds unchanged
    thresholds_after = model.thresholds.detach().cpu().clone()
    threshold_diff = (thresholds_after - thresholds_before).abs().max().item()
    print(f"\nMax threshold change: {threshold_diff:.10f}")
    if threshold_diff > 1e-6:
        print("  WARNING: Thresholds changed unexpectedly!")
    else:
        print("  Thresholds unchanged (as expected)")

    print("\nEvaluating model AFTER post-training...")
    after_train, after_val = evaluate_model(
        model, train_loader, val_loader, image_shape=spike_shape
    )
    print(f"  Train accuracy: {after_train['accuracy']:.4f}")
    print(f"  Val accuracy: {after_val['accuracy']:.4f}")

    model_cpu = model.cpu()
    save_model(model_cpu, os.path.join(output_dir, "model_stdp_weights.pth"))

    plot_weights_evolution(history["weights"], output_dir)
    plot_convergence(history["mean_dw"], output_dir)
    plot_neuron_weights(
        model_cpu.weights, history["spike_counts"], output_dir, image_shape
    )

    all_metrics = {
        "Before": {"train": before_train, "val": before_val},
        "After": {"train": after_train, "val": after_val},
    }
    plot_comparison(all_metrics, output_dir)

    with open(os.path.join(output_dir, "metrics_stdp_weights.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(output_dir, "history_stdp_weights.json"), "w") as f:
        json.dump({"mean_dw": history["mean_dw"]}, f, indent=2)

    print(f"\nSaved post-trained model to {output_dir}/model_stdp_weights.pth")
    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
