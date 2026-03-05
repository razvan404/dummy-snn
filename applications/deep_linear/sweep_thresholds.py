import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset
from spiking import load_model
from spiking.layers import SpikingSequential

DEFAULT_FRACTIONS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DEFAULT_SEEDS = [400, 401, 402]


def sweep_thresholds(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Sweep threshold perturbation magnitudes and return results.

    For each fraction, draws all thresholds from a single distribution centered
    on the original mean threshold, with spread = fraction * mean_threshold.
    """
    if fractions is None:
        fractions = DEFAULT_FRACTIONS
    if seeds is None:
        seeds = DEFAULT_SEEDS

    train_loader, val_loader = dataset_loaders

    # Get baseline
    model = load_model(model_path)
    layer = model.layers[layer_idx]
    mean_thresh = layer.thresholds.mean().item()

    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    _, baseline_metrics = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape
    )
    baseline_acc = baseline_metrics["accuracy"]

    results = {
        "mean_threshold": mean_thresh,
        "baseline_val_accuracy": baseline_acc,
        "seeds": seeds,
        "fractions": [],
    }

    for fraction in fractions:
        perturbation = fraction * mean_thresh
        entry = {
            "fraction": fraction,
            "abs_perturbation": perturbation,
            "normal": {"accuracies": []},
            "uniform": {"accuracies": []},
        }

        for seed in seeds:
            for dist_name in ("normal", "uniform"):
                set_seed(seed)
                model = load_model(model_path)
                layer = model.layers[layer_idx]

                if dist_name == "normal":
                    layer.thresholds.data = torch.normal(
                        mean_thresh, perturbation, size=layer.thresholds.shape
                    ).clamp(min=1.0)
                else:
                    layer.thresholds.data = (
                        torch.empty(layer.thresholds.shape)
                        .uniform_(
                            mean_thresh - perturbation, mean_thresh + perturbation
                        )
                        .clamp(min=1.0)
                    )

                sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
                _, val_m = evaluate_model(
                    sub_model, train_loader, val_loader, spike_shape
                )
                entry[dist_name]["accuracies"].append(val_m["accuracy"])

        for dist_name in ("normal", "uniform"):
            accs = entry[dist_name]["accuracies"]
            entry[dist_name]["mean"] = float(np.mean(accs))
            entry[dist_name]["std"] = float(np.std(accs))

        results["fractions"].append(entry)

    return results


def print_results(results: dict):
    """Print sweep results as a formatted table."""
    mean_thresh = results["mean_threshold"]
    baseline = results["baseline_val_accuracy"]

    print(f"Mean threshold: {mean_thresh:.1f}")
    print(f"Baseline val accuracy: {baseline * 100:.2f}%")
    print()
    print(
        f"{'Fraction':>10}  {'Abs Pert':>10}  {'Normal (val acc)':>18}  {'Uniform (val acc)':>18}"
    )
    print(
        f"{'--------':>10}  {'--------':>10}  {'----------------':>18}  {'-----------------':>18}"
    )

    for entry in results["fractions"]:
        frac_str = f"{entry['fraction'] * 100:.1f}%"
        abs_str = f"{entry['abs_perturbation']:.2f}"
        n = entry["normal"]
        u = entry["uniform"]
        normal_str = f"{n['mean'] * 100:.2f} +/- {n['std'] * 100:.2f}%"
        uniform_str = f"{u['mean'] * 100:.2f} +/- {u['std'] * 100:.2f}%"
        print(f"{frac_str:>10}  {abs_str:>10}  {normal_str:>18}  {uniform_str:>18}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep threshold perturbation magnitudes"
    )
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="Perturbation fractions relative to mean threshold",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for JSON output (default: <model_dir>/threshold_sweep.json)",
    )
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    results = sweep_thresholds(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        layer_idx=args.layer_idx,
        fractions=args.fractions,
        seeds=args.seeds,
    )

    print_results(results)

    output_path = args.output or os.path.join(
        os.path.dirname(args.model_path), "threshold_sweep.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_path}")
