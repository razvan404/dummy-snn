import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from applications.datasets import DATASETS, create_dataset
from spiking import load_model
from spiking.evaluation.eval_classifier import evaluate_classifier
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.layers.integrate_and_fire import IntegrateAndFireLayer

DEFAULT_SCALE_FACTORS = [
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.97,
    0.98,
    1.02,
    1.03,
    1.05,
    1.1,
]


def _extract_features_multi_factor(
    layer: IntegrateAndFireLayer,
    dataloader: DataLoader,
    factors: list[float],
    t_target: float | None = None,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Extract features for multiple weight scale factors in a single data pass.

    Runs the time-step loop once over all samples, computing the weight
    contribution per timestep once and checking all factors via scalar multiply.
    """
    layer.eval()
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    all_times, all_labels = next(iter(full_loader))
    input_times = all_times.flatten(1)
    B = input_times.shape[0]
    O = layer.num_outputs
    thresholds = layer.thresholds
    y = all_labels.numpy()

    results = {
        f: torch.full((B, O), float("inf"), dtype=input_times.dtype) for f in factors
    }
    not_yet_spiked = {f: torch.ones((B, O), dtype=torch.bool) for f in factors}

    with torch.no_grad():
        finite_mask = torch.isfinite(input_times)
        if finite_mask.any():
            unique_times = input_times[finite_mask].unique().sort()[0]
            cum_potential = torch.zeros((B, O), dtype=input_times.dtype)

            for t in unique_times:
                active = (input_times == t).float()
                contrib = active @ layer.weights.T
                cum_potential += contrib

                for factor in factors:
                    crossed = (cum_potential * factor >= thresholds) & not_yet_spiked[
                        factor
                    ]
                    results[factor][crossed] = t
                    not_yet_spiked[factor] &= ~crossed

    return {
        f: (spike_times_to_features(results[f], t_target).numpy(), y) for f in factors
    }


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

    Precomputes cumulative potentials once, then derives all factors' spike times
    analytically (factor * cum_potential >= threshold).

    :returns: {"baseline": metrics_dict, "factors": {0.7: metrics_dict, ...}}
    """
    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS

    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    layer.cpu()

    all_factors = [1.0] + scale_factors

    train_features = _extract_features_multi_factor(
        layer, train_loader, all_factors, t_target=t_target
    )
    val_features = _extract_features_multi_factor(
        layer, val_loader, all_factors, t_target=t_target
    )

    results = {}
    for factor in all_factors:
        X_train, y_train = train_features[factor]
        X_val, y_val = val_features[factor]
        _, val_metrics = evaluate_classifier(X_train, y_train, X_val, y_val)
        results[factor] = val_metrics

    baseline = results.pop(1.0)
    return {"baseline": baseline, "factors": results}


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


def _load_existing_results(output_path: str) -> dict | None:
    """Load existing weight_scaling.json if it exists."""
    if not os.path.exists(output_path):
        return None
    with open(output_path) as f:
        return json.load(f)


def _diff_factors(
    existing: dict | None, desired: list[float]
) -> tuple[list[float], list[str]]:
    """Return (missing factors to compute, stale factor keys to remove)."""
    if existing is None:
        return desired, []
    desired_keys = {str(f) for f in desired}
    existing_keys = set(existing.get("factors", {}).keys())
    missing = [f for f in desired if str(f) not in existing_keys]
    stale = [k for k in existing_keys if k not in desired_keys]
    return missing, stale


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

        existing = None if force else _load_existing_results(output_path)
        missing, stale = _diff_factors(existing, DEFAULT_SCALE_FACTORS)

        if not missing and not stale:
            print(f"  skip {model_path} (already complete)")
            continue

        if existing is not None:
            # Incremental: compute only missing factors, keep existing baseline
            if missing:
                print(
                    f"  scaling {model_path} (t_obj={t_obj}, {len(missing)} new factors)"
                )
                result = weight_scaling_sweep(
                    model_path=model_path,
                    dataset_loaders=(train_loader, val_loader),
                    spike_shape=spike_shape,
                    t_target=t_obj,
                    scale_factors=missing,
                )
                for k, v in result["factors"].items():
                    existing["factors"][str(k)] = v
            if stale:
                print(f"  removing {len(stale)} stale factors from {output_path}")
                for k in stale:
                    del existing["factors"][k]
            serializable = existing
        else:
            print(f"  scaling {model_path} (t_obj={t_obj})")
            result = weight_scaling_sweep(
                model_path=model_path,
                dataset_loaders=(train_loader, val_loader),
                spike_shape=spike_shape,
                t_target=t_obj,
            )
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
