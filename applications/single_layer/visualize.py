import io
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spiking.training import TrainingMonitor


def save_visualizations(
    monitor: TrainingMonitor, figures_dir: str, image_shape: tuple[int, int]
):
    """Save weight evolution, network evolution, and weight visualization plots."""
    plt.figure(figsize=(12, 5))
    plt.suptitle("Weights evolution")
    plt.subplot(1, 2, 1)
    monitor.plot_weight_evolution("train", title="Train split")
    plt.subplot(1, 2, 2)
    monitor.plot_weight_evolution("val", title="Validation split")
    plt.savefig(f"{figures_dir}/weight_evolution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.suptitle("Network evolution")
    plt.subplot(1, 2, 1)
    monitor.plot_thresholds_evolution(title="Thresholds evolution")
    plt.subplot(1, 2, 2)
    monitor.plot_neurons_activity()
    plt.savefig(f"{figures_dir}/network_evolution.png")
    plt.close()

    monitor.visualize_weights(
        image_shape,
        monitor.most_active_neurons(min(32, monitor.model.num_outputs)),
        ncols=8,
    )
    plt.savefig(f"{figures_dir}/weights.png")
    plt.close()


def plot_comparison(all_metrics: dict, output_dir: str):
    """Plot comparison of before/after metrics as grouped bar charts."""
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
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=150)
    plt.close()


def create_histogram_frame(
    thresholds, init_threshold_mean, step, bins=30, figsize=(10, 6)
):
    """Create a single histogram frame for the threshold GIF."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(thresholds, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Threshold Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Threshold Distribution - Step {step}", fontsize=14)
    plt.suptitle(f"Init threshold mean: {init_threshold_mean}")
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {np.mean(thresholds):.3f}\nStd: {np.std(thresholds):.3f}"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").copy()
    buf.close()
    plt.close(fig)

    return img


def _create_frame_worker(args):
    """Worker function for parallel frame creation."""
    step, thresholds, init_threshold_mean, bins = args
    return create_histogram_frame(thresholds, init_threshold_mean, step, bins)


def create_threshold_gif(
    json_path, output_path, setup_path, duration=200, bins=30, num_workers=None
):
    """Create an animated GIF from threshold evolution data.

    Args:
        json_path: Path to thresholds.json file
        output_path: Path where GIF will be saved
        setup_path: Path to setup.json file
        duration: Duration of each frame in milliseconds
        bins: Number of bins for histograms
        num_workers: Number of parallel workers (default: cpu_count())
    """
    import tqdm

    if num_workers is None:
        num_workers = cpu_count()

    print(f"Loading thresholds from {json_path}...")
    with open(json_path, "r") as f:
        threshold_lists = json.load(f)
    with open(setup_path, "r") as f:
        init_threshold_mean = json.load(f)["threshold_init"]["avg_threshold"]

    # Sample every 5th step
    sampled_thresholds = threshold_lists[::5]

    print(
        f"Creating {len(sampled_thresholds)} histogram frames using {num_workers} workers..."
    )

    # Prepare arguments for parallel processing
    args_list = [
        (step, thresholds, init_threshold_mean, bins)
        for step, thresholds in enumerate(sampled_thresholds)
    ]

    # Create frames in parallel with progress bar
    with Pool(num_workers) as pool:
        frames = list(
            tqdm.tqdm(pool.imap(_create_frame_worker, args_list), total=len(args_list))
        )

    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )

    print(f"Done! Created GIF with {len(frames)} frames.")


if __name__ == "__main__":
    # Default paths for threshold GIF creation
    log_dir = (
        Path(__file__).parent.parent.parent / "logs" / "model_th55.6_seed42_outs100"
    )
    json_path = log_dir / "thresholds.json"
    setup_path = log_dir / "setup.json"
    output_path = log_dir / "thresholds_evolution.gif"

    create_threshold_gif(json_path, output_path, setup_path, duration=50, bins=20)
