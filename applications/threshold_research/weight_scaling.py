import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import DATASETS, create_dataset
from spiking import load_model
from spiking.layers import SpikingSequential

DEFAULT_SCALE_FACTORS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1]


def weight_scaling_sweep(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    scale_factors: list[float] | None = None,
    t_target: float | None = None,
) -> dict:
    """Scale layer weights by various factors and evaluate each.

    Returns: {"baseline": metrics_dict, "factors": {0.7: metrics_dict, ...}}
    """
    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS

    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    original_weights = layer.weights.detach().clone()
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    _, baseline = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape, t_target=t_target
    )

    factors = {}
    for factor in scale_factors:
        layer.weights.data.copy_(original_weights * factor)
        _, val_metrics = evaluate_model(
            sub_model, train_loader, val_loader, spike_shape, t_target=t_target
        )
        factors[factor] = val_metrics

    # Restore original weights
    layer.weights.data.copy_(original_weights)

    return {"baseline": baseline, "factors": factors}


def _find_trained_models(base_dir: str) -> list[tuple[str, float]]:
    """Find model.pth files and their t_objectives under base_dir."""
    models = []
    if not os.path.isdir(base_dir):
        return models
    for tobj_dir in sorted(os.listdir(base_dir)):
        if not tobj_dir.startswith("tobj_"):
            continue
        t_obj = float(tobj_dir.split("_")[1])
        tobj_path = os.path.join(base_dir, tobj_dir)
        for seed_dir in sorted(os.listdir(tobj_path)):
            if not seed_dir.startswith("seed_"):
                continue
            model_path = os.path.join(tobj_path, seed_dir, "model.pth")
            if os.path.exists(model_path):
                models.append((model_path, t_obj))
    return models


def run(dataset: str, *, force: bool = False):
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_trained_models(base_dir)
    if not models:
        print(f"No models found under {base_dir}")
        return

    for model_path, t_obj in models:
        output_path = os.path.join(os.path.dirname(model_path), "weight_scaling.json")
        if not force and os.path.exists(output_path):
            print(f"  skip {model_path} (already complete)")
            continue

        print(f"  scaling {model_path} (t_obj={t_obj})")
        result = weight_scaling_sweep(
            model_path=model_path,
            dataset_loaders=(train_loader, val_loader),
            spike_shape=spike_shape,
            t_target=t_obj,
        )

        # Convert float keys to strings for JSON
        serializable = {
            "baseline": result["baseline"],
            "factors": {str(k): v for k, v in result["factors"].items()},
        }
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Weight scaling experiment")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    args = parser.parse_args()
    run(args.dataset, force=args.force)
