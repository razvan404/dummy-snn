import json
import os
import re

import numpy as np
import torch

from applications.pipeline.datasets import load_split_data  # re-exported
from applications.pipeline.layout import RunSpec
from spiking.evaluation import evaluate_classifier, extract_features

__all__ = [
    "set_seed",
    "resolve_model_dir",
    "resolve_params",
    "load_split_data",
    "evaluate_model",
    "aggregate_metrics",
    "merge_seed_results",
]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def resolve_model_dir(dataset: str, num_filters: int, t_obj: float, seed: int) -> str:
    return str(RunSpec.single(dataset, num_filters, t_obj, seed).model_dir)


def resolve_params(args) -> tuple[int, float, str]:
    from applications.paper_hyperparams import get_paper_hyperparams

    hp = get_paper_hyperparams(args.dataset)
    nf = args.num_filters or hp["num_filters"]
    t_obj = args.t_obj if args.t_obj is not None else hp["target_timestamp"]
    return nf, t_obj, resolve_model_dir(args.dataset, nf, t_obj, args.seed)


def evaluate_model(model, train_loader, val_loader, t_target=None):
    model = model.cpu()
    X_train, y_train = extract_features(model, train_loader, t_target)
    X_test, y_test = extract_features(model, val_loader, t_target)
    return evaluate_classifier(X_train, y_train, X_test, y_test)


def aggregate_metrics(all_metrics: list[dict]) -> dict:
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
