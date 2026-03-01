import json
import os
import re

import numpy as np
import torch

from spiking.evaluation import extract_features, evaluate_classifier


def set_seed(seed: int):
    """Set numpy + torch + cuda seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_model(model, train_loader, val_loader, image_shape):
    """Extract features and evaluate classifier. Moves model to CPU.

    image_shape: e.g. (2, 16, 16) — the full spike volume shape.
    """
    model = model.cpu()
    X_train, y_train = extract_features(model, train_loader, image_shape)
    X_test, y_test = extract_features(model, val_loader, image_shape)
    return evaluate_classifier(X_train, y_train, X_test, y_test)


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean and std for each metric across seeds.

    all_metrics: list of {"train": {...}, "validation": {...}} dicts.
    Returns: {"train": {"accuracy": {"mean": ..., "std": ...}, ...}, ...}
    """
    splits = all_metrics[0].keys()
    metric_keys = all_metrics[0][next(iter(splits))].keys()

    summary = {}
    for split in splits:
        summary[split] = {}
        for key in metric_keys:
            values = [m[split][key] for m in all_metrics]
            summary[split][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def merge_seed_results(directory: str):
    """Scan seed_*/metrics.json under directory, produce merged_results.json and summary.json.

    merged_results.json: {"seeds": [1,2,...], "train": {"accuracy": [...], ...}, "validation": {...}}
    summary.json: mean/std per metric via aggregate_metrics().
    """
    seed_dirs = []
    for name in os.listdir(directory):
        match = re.match(r"^seed_(\d+)$", name)
        if match and os.path.isdir(os.path.join(directory, name)):
            seed_dirs.append((int(match.group(1)), name))
    seed_dirs.sort(key=lambda x: x[0])

    all_metrics = []
    seeds = []
    for seed_num, dirname in seed_dirs:
        metrics_path = os.path.join(directory, dirname, "metrics.json")
        with open(metrics_path) as f:
            all_metrics.append(json.load(f))
        seeds.append(seed_num)

    # Build merged: {"seeds": [...], "train": {"accuracy": [...], ...}, ...}
    splits = all_metrics[0].keys()
    metric_keys = all_metrics[0][next(iter(splits))].keys()
    merged = {"seeds": seeds}
    for split in splits:
        merged[split] = {}
        for key in metric_keys:
            merged[split][key] = [m[split][key] for m in all_metrics]

    with open(os.path.join(directory, "merged_results.json"), "w") as f:
        json.dump(merged, f, indent=4)

    summary = aggregate_metrics(all_metrics)
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
