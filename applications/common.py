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
