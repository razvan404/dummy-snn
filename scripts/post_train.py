import argparse
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.common import set_seed, evaluate_model, create_dataset
from spiking.learning import Learner
from spiking.threshold import PlasticityBalanceAdaptation
from spiking.training import UnsupervisedTrainer
from spiking.utils import load_model, save_model

IMAGE_SHAPE = (16, 16)


def plot_thresholds_evolution(
    thresholds_history: list[torch.Tensor],
    output_dir: str,
):
    """Plot threshold evolution over epochs."""
    thresholds_array = np.array([t.numpy() for t in thresholds_history])
    num_epochs, num_neurons = thresholds_array.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i in range(num_neurons):
        ax.plot(thresholds_array[:, i], alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold Evolution (All Neurons)")

    ax = axes[0, 1]
    mean_th = thresholds_array.mean(axis=1)
    std_th = thresholds_array.std(axis=1)
    epochs = np.arange(num_epochs)
    ax.plot(epochs, mean_th, label="Mean", color="blue")
    ax.fill_between(epochs, mean_th - std_th, mean_th + std_th, alpha=0.3, color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Threshold")
    ax.set_title("Mean Threshold ± Std")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(thresholds_array[0], bins=30, alpha=0.5, label="Start", color="blue")
    ax.hist(thresholds_array[-1], bins=30, alpha=0.5, label="End", color="orange")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.set_title("Threshold Distribution: Start vs End")
    ax.legend()

    ax = axes[1, 1]
    sorted_indices = np.argsort(thresholds_array[-1])
    sorted_thresholds = thresholds_array[:, sorted_indices].T
    im = ax.imshow(sorted_thresholds, aspect="auto", cmap="viridis")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Neuron (sorted by final threshold)")
    ax.set_title("Threshold Heatmap")
    plt.colorbar(im, ax=ax, label="Threshold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thresholds_evolution.png"), dpi=150)
    plt.close()


def plot_convergence(mean_deltas: list[float], output_dir: str):
    """Plot convergence of threshold updates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_deltas, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean |Threshold Delta|")
    ax.set_title("Convergence of Threshold Updates")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
    plt.close()


def plot_comparison(all_metrics: dict, output_dir: str):
    """Plot comparison of before/after metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metrics_names = ["accuracy", "precision", "recall", "f1"]
    config_names = list(all_metrics.keys())
    colors = ["steelblue", "darkorange"]
    x = np.arange(len(metrics_names))
    width = 0.35

    ax = axes[0]
    for idx, config in enumerate(config_names):
        vals = [all_metrics[config]["train"][m] for m in metrics_names]
        offset = (idx - 0.5) * width
        ax.bar(x + offset, vals, width, label=config, color=colors[idx])
        for i, v in enumerate(vals):
            ax.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Train Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.15)

    ax = axes[1]
    for idx, config in enumerate(config_names):
        vals = [all_metrics[config]["val"][m] for m in metrics_names]
        offset = (idx - 0.5) * width
        ax.bar(x + offset, vals, width, label=config, color=colors[idx])
        for i, v in enumerate(vals):
            ax.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"), dpi=150)
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Post-train thresholds using STDP balance"
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
        help="Maximum number of epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for threshold updates (default: 0.1)",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=1.0,
        help="Minimum threshold value (default: 1.0)",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=100.0,
        help="Maximum threshold value (default: 100.0)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.001,
        help="Stop when mean |delta| falls below this (default: 0.001)",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_model(args.model_path)
    print(f"Loaded model from {args.model_path}")
    print(f"Model has {model.num_outputs} neurons")

    output_dir = args.output_dir or os.path.dirname(args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader = create_dataset("mnist_subset", IMAGE_SHAPE)

    print("\nEvaluating model BEFORE post-training...")
    before_train, before_val = evaluate_model(
        copy.deepcopy(model), train_loader, val_loader, image_shape=(2, *IMAGE_SHAPE)
    )
    print(f"  Train accuracy: {before_train['accuracy']:.4f}")
    print(f"  Val accuracy: {before_val['accuracy']:.4f}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    adaptation = PlasticityBalanceAdaptation(
        learning_rate=args.learning_rate,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
    )
    learner = Learner(model, learning_mechanism=None, threshold_adaptation=adaptation)
    trainer = UnsupervisedTrainer(
        model=model,
        learner=learner,
        image_shape=(2, *IMAGE_SHAPE),
        early_stopping=False,
    )

    print(f"\nRunning STDP threshold optimization...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Threshold range: [{args.min_threshold}, {args.max_threshold}]")

    history = {
        "mean_delta": [],
        "thresholds": [model.thresholds.detach().cpu().clone()],
    }

    for epoch in range(args.num_epochs):
        thresholds_before = model.thresholds.detach().clone()

        trainer.step_loader(train_loader, split="train")

        threshold_deltas = (model.thresholds - thresholds_before).abs()
        mean_delta = threshold_deltas.mean().item()
        history["mean_delta"].append(mean_delta)
        history["thresholds"].append(model.thresholds.detach().cpu().clone())

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"mean |delta| = {mean_delta:.6f}"
        )

        if mean_delta < args.convergence_threshold:
            print(f"Converged at epoch {epoch + 1}")
            break

    print("\nEvaluating model AFTER post-training...")
    after_train, after_val = evaluate_model(model, train_loader, val_loader, image_shape=(2, *IMAGE_SHAPE))
    print(f"  Train accuracy: {after_train['accuracy']:.4f}")
    print(f"  Val accuracy: {after_val['accuracy']:.4f}")

    model_cpu = model.cpu()
    save_model(model_cpu, os.path.join(output_dir, "model_post_trained.pth"))

    plot_thresholds_evolution(history["thresholds"], output_dir)
    plot_convergence(history["mean_delta"], output_dir)

    all_metrics = {
        "Before": {"train": before_train, "val": before_val},
        "After": {"train": after_train, "val": after_val},
    }
    plot_comparison(all_metrics, output_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({"mean_delta": history["mean_delta"]}, f, indent=2)

    print(f"\nSaved post-trained model to {output_dir}/model_post_trained.pth")
    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
