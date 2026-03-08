import json
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from torch.utils.data import DataLoader

from applications.common import set_seed
from spiking import load_model
from spiking.evaluation.feature_extraction import (
    spike_times_to_features,
    extract_features,
)
from spiking.evaluation.eval_utils import compute_metrics
from spiking.layers import SpikingSequential


def precompute_cumulative_potentials(
    input_times: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cumulative membrane potentials at each unique input time boundary.

    input_times: (B, I) — spike times for each input, inf means no spike.
    weights: (O, I) — layer weights.

    Returns:
        cum_potentials: (B, O, G) — cumulative potential at each boundary time.
        boundary_times: (G,) — sorted unique finite input times.
    """
    B, I = input_times.shape
    O = weights.shape[0]

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return torch.zeros((B, O, 0)), torch.zeros(0)

    boundary_times = input_times[finite_mask].unique().sort()[0]
    G = len(boundary_times)

    cum_potentials = torch.zeros((B, O, G), dtype=input_times.dtype)
    running = torch.zeros((B, O), dtype=input_times.dtype)

    for g, t in enumerate(boundary_times):
        active = (input_times == t).float()  # (B, I)
        contrib = active @ weights.T  # (B, O)
        running = running + contrib
        cum_potentials[:, :, g] = running

    return cum_potentials, boundary_times


def spike_times_from_potentials(
    cum_potentials: torch.Tensor,
    boundary_times: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Find first threshold crossing time from precomputed cumulative potentials.

    cum_potentials: (B, G) — cumulative potential for one neuron across boundary times.
    boundary_times: (G,) — sorted unique times.
    threshold: scalar threshold value.

    Returns: (B,) — spike time for each sample (inf if no crossing).
    """
    B = cum_potentials.shape[0]
    if cum_potentials.shape[1] == 0:
        return torch.full((B,), float("inf"), dtype=cum_potentials.dtype)

    crossed = cum_potentials >= threshold  # (B, G)
    any_crossed = crossed.any(dim=1)  # (B,)
    first_crossing = crossed.float().argmax(dim=1)  # (B,)

    result = torch.full((B,), float("inf"), dtype=cum_potentials.dtype)
    result[any_crossed] = boundary_times[first_crossing[any_crossed]]

    return result


def _collect_input_times(loader):
    """Collect all input spike times from a DataLoader into a single tensor."""
    batched = DataLoader(loader.dataset, batch_size=256, shuffle=False)
    parts = []
    for batch_times, _labels in batched:
        parts.append(batch_times.flatten(1))
    return torch.cat(parts, dim=0)


def _compute_fraction_features(cum_potentials, boundary_times, thresholds, t_target):
    """Compute feature matrix for all neurons at a given set of thresholds."""
    n_samples = cum_potentials.shape[0]
    num_outputs = cum_potentials.shape[1]
    spike_times = torch.full(
        (n_samples, num_outputs), float("inf"), dtype=cum_potentials.dtype
    )
    for neuron_idx in range(num_outputs):
        spike_times[:, neuron_idx] = spike_times_from_potentials(
            cum_potentials[:, neuron_idx, :],
            boundary_times,
            thresholds[neuron_idx].item(),
        )
    return spike_times_to_features(spike_times, t_target).numpy()


# --- Cache helpers ---


def _feature_cache_exists(cache_dir):
    return os.path.exists(os.path.join(cache_dir, "metadata.json"))


def _save_feature_cache(cache_dir, features):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "baseline_train.npy"), features["baseline_train"])
    np.save(os.path.join(cache_dir, "baseline_val.npy"), features["baseline_val"])
    np.save(os.path.join(cache_dir, "labels_train.npy"), features["labels_train"])
    np.save(os.path.join(cache_dir, "labels_val.npy"), features["labels_val"])
    np.save(os.path.join(cache_dir, "perturbed_train.npy"), features["perturbed_train"])
    np.save(os.path.join(cache_dir, "perturbed_val.npy"), features["perturbed_val"])
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "original_thresholds": features["original_thresholds"],
                "perturbation_fractions": features["perturbation_fractions"],
            },
            f,
        )


def _load_feature_cache(cache_dir):
    with open(os.path.join(cache_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return {
        "baseline_train": np.load(os.path.join(cache_dir, "baseline_train.npy")),
        "baseline_val": np.load(os.path.join(cache_dir, "baseline_val.npy")),
        "labels_train": np.load(os.path.join(cache_dir, "labels_train.npy")),
        "labels_val": np.load(os.path.join(cache_dir, "labels_val.npy")),
        "perturbed_train": np.load(os.path.join(cache_dir, "perturbed_train.npy")),
        "perturbed_val": np.load(os.path.join(cache_dir, "perturbed_val.npy")),
        "original_thresholds": metadata["original_thresholds"],
        "perturbation_fractions": metadata["perturbation_fractions"],
    }


def _save_partial_results(cache_dir, completed_fractions, accuracy_matrix, f1_matrix):
    os.makedirs(cache_dir, exist_ok=True)
    sorted_completed = sorted(completed_fractions)
    partial = {
        "completed_fractions": sorted_completed,
        "accuracy_matrix": accuracy_matrix[:, sorted_completed].tolist(),
        "f1_matrix": f1_matrix[:, sorted_completed].tolist(),
    }
    with open(os.path.join(cache_dir, "partial_results.json"), "w") as f:
        json.dump(partial, f)


def _load_partial_results(cache_dir):
    path = os.path.join(cache_dir, "partial_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --- Phase A: Inference (cacheable) ---


def compute_perturbed_features(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    t_target: float | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> dict:
    """Compute baseline and perturbed feature matrices for all fractions.

    If cache_dir is provided, saves results to disk and loads from cache on
    subsequent calls.
    """
    if cache_dir and _feature_cache_exists(cache_dir):
        return _load_feature_cache(cache_dir)

    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    sub_model = sub_model.cpu()

    num_outputs = layer.num_outputs
    original_thresholds = layer.thresholds.detach().clone()
    perturbation_fractions = [round(-0.5 + i * 0.025, 3) for i in range(31)]

    # Extract unperturbed features as baseline
    X_train, y_train = extract_features(sub_model, train_loader, spike_shape, t_target)
    X_val, y_val = extract_features(sub_model, val_loader, spike_shape, t_target)

    # Precompute cumulative potentials for both train and val
    weights = layer.weights.detach()
    train_input_times = _collect_input_times(train_loader)
    val_input_times = _collect_input_times(val_loader)

    train_cum, train_boundary = precompute_cumulative_potentials(
        train_input_times, weights
    )
    val_cum, val_boundary = precompute_cumulative_potentials(val_input_times, weights)

    # Compute perturbed features for all fractions
    num_fracs = len(perturbation_fractions)
    perturbed_train = np.zeros(
        (num_fracs, X_train.shape[0], num_outputs), dtype=np.float32
    )
    perturbed_val = np.zeros((num_fracs, X_val.shape[0], num_outputs), dtype=np.float32)

    for frac_idx, frac in enumerate(perturbation_fractions):
        new_thresholds = original_thresholds * (1.0 + frac)
        perturbed_train[frac_idx] = _compute_fraction_features(
            train_cum, train_boundary, new_thresholds, t_target
        )
        perturbed_val[frac_idx] = _compute_fraction_features(
            val_cum, val_boundary, new_thresholds, t_target
        )

    features = {
        "baseline_train": X_train,
        "baseline_val": X_val,
        "labels_train": y_train,
        "labels_val": y_val,
        "perturbed_train": perturbed_train,
        "perturbed_val": perturbed_val,
        "original_thresholds": original_thresholds.tolist(),
        "perturbation_fractions": perturbation_fractions,
    }

    if cache_dir:
        _save_feature_cache(cache_dir, features)

    return features


# --- Phase B: Evaluation (resumable) ---


def evaluate_perturbations(
    *,
    features: dict,
    cache_dir: str | None = None,
) -> dict:
    """Run per-neuron classifier fits on precomputed features.

    If cache_dir is provided, saves results incrementally after each fraction
    and resumes from partial results on restart.
    """
    X_train = features["baseline_train"]
    X_val = features["baseline_val"]
    y_train = features["labels_train"]
    y_val = features["labels_val"]
    perturbed_train = features["perturbed_train"]
    perturbed_val = features["perturbed_val"]
    original_thresholds = features["original_thresholds"]
    perturbation_fractions = features["perturbation_fractions"]

    num_outputs = X_train.shape[1]
    num_fracs = len(perturbation_fractions)

    # Baseline classifier
    baseline_clf = RidgeClassifier()
    baseline_clf.fit(X_train, y_train)
    baseline_metrics = compute_metrics(y_val, baseline_clf.predict(X_val))

    # Load partial results if resuming
    accuracy_matrix = np.zeros((num_outputs, num_fracs))
    f1_matrix = np.zeros((num_outputs, num_fracs))
    completed_fractions = set()

    if cache_dir:
        partial = _load_partial_results(cache_dir)
        if partial:
            completed = partial["completed_fractions"]
            completed_fractions = set(completed)
            acc_arr = np.array(partial["accuracy_matrix"])
            f1_arr = np.array(partial["f1_matrix"])
            for i, frac_idx in enumerate(completed):
                accuracy_matrix[:, frac_idx] = acc_arr[:, i]
                f1_matrix[:, frac_idx] = f1_arr[:, i]

    for frac_idx in range(num_fracs):
        if frac_idx in completed_fractions:
            continue

        # For each neuron, swap that column and retrain
        for neuron_idx in range(num_outputs):
            X_train_mod = X_train.copy()
            X_train_mod[:, neuron_idx] = perturbed_train[frac_idx, :, neuron_idx]
            X_val_mod = X_val.copy()
            X_val_mod[:, neuron_idx] = perturbed_val[frac_idx, :, neuron_idx]

            clf = RidgeClassifier()
            clf.fit(X_train_mod, y_train)
            y_pred = clf.predict(X_val_mod)
            metrics = compute_metrics(y_val, y_pred)
            accuracy_matrix[neuron_idx, frac_idx] = metrics["accuracy"]
            f1_matrix[neuron_idx, frac_idx] = metrics["f1"]

        completed_fractions.add(frac_idx)
        if cache_dir:
            _save_partial_results(
                cache_dir, completed_fractions, accuracy_matrix, f1_matrix
            )

    # Find per-neuron optimal perturbation
    best_frac_indices = accuracy_matrix.argmax(axis=1)
    optimal_fracs = [perturbation_fractions[i] for i in best_frac_indices]
    optimal_thresholds = [
        (original_thresholds[n] * (1.0 + optimal_fracs[n])) for n in range(num_outputs)
    ]
    optimal_deltas = [
        optimal_thresholds[n] - original_thresholds[n] for n in range(num_outputs)
    ]

    return {
        "baseline": baseline_metrics,
        "original_thresholds": original_thresholds,
        "perturbation_fractions": perturbation_fractions,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "f1_matrix": f1_matrix.tolist(),
        "optimal_thresholds": optimal_thresholds,
        "optimal_deltas": optimal_deltas,
    }


# --- Top-level entry point ---


def run_perturbation_sweep(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    t_target: float | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> dict:
    """Run per-neuron threshold perturbation sweep.

    For each neuron, perturbs its threshold by fractions from -0.5 to +0.25 (step 0.025),
    recomputes only that neuron's features, and measures accuracy impact.

    Returns dict with baseline metrics, perturbation matrices, and optimal thresholds.
    """
    features = compute_perturbed_features(
        model_path=model_path,
        dataset_loaders=dataset_loaders,
        spike_shape=spike_shape,
        layer_idx=layer_idx,
        t_target=t_target,
        seed=seed,
        cache_dir=cache_dir,
    )
    return evaluate_perturbations(features=features, cache_dir=cache_dir)
