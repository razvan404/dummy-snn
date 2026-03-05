import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import DATASETS, create_dataset
from spiking import load_model
from spiking.layers import SpikingSequential


def evaluate_optimal_thresholds(
    *,
    model_path: str,
    perturbation_results: dict,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    t_target: float | None = None,
    seed: int = 42,
) -> dict:
    """Set all neurons to their per-neuron optimal thresholds and evaluate.

    Returns dict with baseline metrics, optimal metrics, and threshold details.
    """
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    original_thresholds = layer.thresholds.detach().clone()

    # Evaluate baseline
    _, baseline_metrics = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape, t_target=t_target
    )

    # Apply per-neuron optimal thresholds
    optimal = torch.tensor(
        perturbation_results["optimal_thresholds"], dtype=layer.thresholds.dtype
    )
    layer.thresholds.data.copy_(optimal)

    _, optimal_metrics = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape, t_target=t_target
    )

    # Restore
    layer.thresholds.data.copy_(original_thresholds)

    return {
        "baseline": baseline_metrics,
        "optimal_combined": optimal_metrics,
        "original_thresholds": original_thresholds.tolist(),
        "optimal_thresholds": perturbation_results["optimal_thresholds"],
        "optimal_deltas": perturbation_results["optimal_deltas"],
        "accuracy_improvement": (
            optimal_metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
    }


def run(dataset: str, *, force: bool = False):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    base_dir = f"logs/{dataset}/threshold_research"
    if not os.path.isdir(base_dir):
        print(f"No results found under {base_dir}")
        return

    for tobj_dir in sorted(os.listdir(base_dir)):
        if not tobj_dir.startswith("tobj_"):
            continue
        t_obj = float(tobj_dir.split("_")[1])
        tobj_path = os.path.join(base_dir, tobj_dir)

        for seed_dir in sorted(os.listdir(tobj_path)):
            if not seed_dir.startswith("seed_"):
                continue
            seed = int(seed_dir.split("_")[1])
            seed_path = os.path.join(tobj_path, seed_dir)

            perturbation_path = os.path.join(seed_path, "perturbation_results.json")
            output_path = os.path.join(seed_path, "optimal_thresholds.json")

            if not os.path.exists(perturbation_path):
                continue
            if not force and os.path.exists(output_path):
                print(f"  skip t_obj={t_obj} seed={seed} (already complete)")
                continue

            model_path = os.path.join(seed_path, "model.pth")
            with open(perturbation_path) as f:
                perturbation_results = json.load(f)

            print(f"  evaluating optimal thresholds t_obj={t_obj} seed={seed}")
            result = evaluate_optimal_thresholds(
                model_path=model_path,
                perturbation_results=perturbation_results,
                dataset_loaders=(train_loader, val_loader),
                spike_shape=spike_shape,
                t_target=t_obj,
                seed=seed,
            )

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

            print(
                f"    baseline={result['baseline']['accuracy']:.4f} "
                f"optimal={result['optimal_combined']['accuracy']:.4f} "
                f"delta={result['accuracy_improvement']:+.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3b: Evaluate combined optimal thresholds"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    args = parser.parse_args()
    run(args.dataset, force=args.force)
