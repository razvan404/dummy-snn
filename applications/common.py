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


def resolve_model_dir(
    dataset: str, num_filters: int, t_obj: float, seed: int
) -> str:
    """Compute model directory from experiment parameters."""
    base = "cifar10_whitened" if dataset == "cifar10" else dataset
    return f"logs/{base}/sweep/nf_{num_filters}/tobj_{t_obj:.2f}/seed_{seed}"


def resolve_params(args) -> tuple[int, float, str]:
    """Return (num_filters, t_obj, model_dir) from CLI args + paper defaults."""
    from applications.paper_hyperparams import get_paper_hyperparams

    hp = get_paper_hyperparams(args.dataset)
    nf = args.num_filters or hp["num_filters"]
    t_obj = args.t_obj if args.t_obj is not None else hp["target_timestamp"]
    return nf, t_obj, resolve_model_dir(args.dataset, nf, t_obj, args.seed)


def load_split_data(dataset: str) -> tuple[dict, dict]:
    """Load train and test image tensors + labels."""
    if dataset == "cifar10":
        from applications.datasets import Cifar10WhitenedDataset

        train_ds = Cifar10WhitenedDataset("data", "train")
        test_ds = Cifar10WhitenedDataset(
            "data", "test", kernels=train_ds.kernels, mean=train_ds.mean
        )
        return (
            {"images": train_ds.all_times, "labels": train_ds.outputs},
            {"images": test_ds.all_times, "labels": test_ds.outputs},
        )
    processed_dir = f"data/processed-{dataset}"
    return (
        torch.load(f"{processed_dir}/train.pt", weights_only=True),
        torch.load(f"{processed_dir}/test.pt", weights_only=True),
    )


def create_dataloaders(dataset: str):
    """Create (train_loader, val_loader) for threshold optimization."""
    if dataset == "cifar10":
        from applications.datasets import Cifar10WhitenedDataset
        from torch.utils.data import DataLoader

        train_ds = Cifar10WhitenedDataset("data", "train")
        val_ds = Cifar10WhitenedDataset(
            "data", "test", kernels=train_ds.kernels, mean=train_ds.mean
        )
        return (
            DataLoader(train_ds, batch_size=None, shuffle=False),
            DataLoader(val_ds, batch_size=None, shuffle=False),
        )
    from applications.datasets import create_dataset

    return create_dataset(dataset)


def evaluate_model(model, train_loader, val_loader, t_target=None):
    """Extract features and evaluate classifier. Moves model to CPU.

    t_target: if provided, use Falez Eq 10 for feature conversion.
    """
    model = model.cpu()

    X_train, y_train = extract_features(model, train_loader, t_target)
    X_test, y_test = extract_features(model, val_loader, t_target)

    return evaluate_classifier(X_train, y_train, X_test, y_test)


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Compute mean and std for each metric across seeds.

    :param all_metrics: List of {"train": {...}, "validation": {...}} dicts.
    :returns: {"train": {"accuracy": {"mean": ..., "std": ...}, ...}, ...}
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
