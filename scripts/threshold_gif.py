import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import io
import tqdm
from multiprocessing import Pool, cpu_count


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def create_histogram_frame(
    thresholds, init_threshold_mean, step, bins=30, figsize=(10, 6)
):
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
    """
    Create an animated GIF from threshold evolution data.

    Args:
        json_path: Path to thresholds.json file
        output_path: Path where GIF will be saved
        setup_path: Path to setup.json file
        duration: Duration of each frame in milliseconds
        bins: Number of bins for histograms
        num_workers: Number of parallel workers (default: cpu_count())
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Loading thresholds from {json_path}...")
    threshold_lists = load_json(json_path)
    init_threshold_mean = load_json(setup_path)["threshold.initialization"][1][
        "avg_threshold"
    ]

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


def main():
    # Define paths
    log_dir = Path(__file__).parent.parent / "logs" / "model_th55.6_seed42_outs100"
    json_path = log_dir / "thresholds.json"
    setup_path = log_dir / "setup.json"
    output_path = log_dir / "thresholds_evolution.gif"

    # Create the GIF
    create_threshold_gif(json_path, output_path, setup_path, duration=50, bins=20)


if __name__ == "__main__":
    main()
