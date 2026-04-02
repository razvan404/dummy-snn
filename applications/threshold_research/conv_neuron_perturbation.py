from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from applications.common import set_seed
from spiking import load_model
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.evaluation.eval_utils import compute_metrics
from spiking.layers import SpikingSequential
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer

from applications.threshold_research.neuron_perturbation import (
    _feature_cache_exists,
    _save_feature_cache,
    _load_feature_cache,
    _save_partial_results,
    _load_partial_results,
)
from applications.threshold_research.perturbation_params import PERTURBATION_FRACTIONS


def collect_conv_input_times(
    loader: DataLoader,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect input spike times preserving spatial dims, plus labels.

    Returns:
        (times, labels): times is (N, C, H, W), labels is (N,).
    """
    batched = DataLoader(loader.dataset, batch_size=chunk_size, shuffle=False)
    time_parts = []
    label_parts = []
    for batch_times, batch_labels in batched:
        time_parts.append(batch_times)
        label_parts.append(batch_labels)
    return torch.cat(time_parts, dim=0), torch.cat(label_parts, dim=0)


@torch.no_grad()
def multi_threshold_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int,
    padding: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Single-pass conv2d accumulation checking multiple threshold sets.

    Args:
        input_times: (B, C, H, W) spike times (inf = no spike).
            Assumed to be already discretized (e.g. via discretize_times with
            num_bins=16/64), so the number of unique times is small.
        weights_4d: (F, C, kH, kW) conv filter weights.
        thresholds_2d: (num_fracs, F) threshold values per fraction.
        stride: Conv stride.
        padding: Conv padding.
        device: Device for computation.

    Returns:
        (num_fracs, B, F, oH, oW) spike times tensor.
    """
    input_times = input_times.to(device)
    weights_4d = weights_4d.to(device)
    thresholds_2d = thresholds_2d.to(device)

    B, C, H, W = input_times.shape
    num_fracs, num_filters = thresholds_2d.shape
    kernel_size = weights_4d.shape[2]
    oH = (H + 2 * padding - kernel_size) // stride + 1
    oW = (W + 2 * padding - kernel_size) // stride + 1

    result = torch.full(
        (num_fracs, B, num_filters, oH, oW),
        float("inf"),
        dtype=input_times.dtype,
        device=device,
    )
    not_yet_spiked = torch.ones(
        (num_fracs, B, num_filters, oH, oW),
        dtype=torch.bool,
        device=device,
    )
    cum_potential = torch.zeros(
        (B, num_filters, oH, oW),
        dtype=input_times.dtype,
        device=device,
    )
    thresholds_5d = thresholds_2d.view(num_fracs, 1, num_filters, 1, 1)

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return result.cpu()

    unique_times = input_times[finite_mask].unique().sort()[0]

    for t in unique_times:
        active = (input_times == t).float()
        contrib = F.conv2d(active, weights_4d, stride=stride, padding=padding)
        cum_potential += contrib

        crossed = (cum_potential.unsqueeze(0) >= thresholds_5d) & not_yet_spiked
        result[crossed] = t
        not_yet_spiked &= ~crossed

        if not not_yet_spiked.any():
            break

    return result.cpu()


def _spike_times_to_pooled_features(
    spike_times: torch.Tensor,
    t_target: float | None,
    pool_size: int,
) -> np.ndarray:
    """Convert (num_fracs, B, F, oH, oW) spike times to flat pooled features.

    Returns: (num_fracs, B, F * pool_size * pool_size) numpy array.
    """
    num_fracs, B, F_dim, oH, oW = spike_times.shape
    # Reshape to (num_fracs*B, F, oH, oW) for batch processing
    flat = spike_times.reshape(num_fracs * B, F_dim, oH, oW)
    features = spike_times_to_features(flat, t_target)
    pooled = sum_pool_features(features, pool_size)
    flat_features = pooled.flatten(1).numpy()
    # Reshape back to (num_fracs, B, flat_dim)
    return flat_features.reshape(num_fracs, B, -1)


def compute_conv_perturbed_features(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    layer_idx: int = 0,
    t_target: float | None = None,
    pool_size: int = 2,
    seed: int = 42,
    cache_dir: str | None = None,
    force: bool = False,
    device: str = "cpu",
    chunk_size: int = 64,
) -> dict:
    """Compute baseline and perturbed feature matrices for a conv layer.

    Uses multi-threshold single-pass inference: conv2d accumulation is done once
    per chunk, all 31 perturbation thresholds are checked simultaneously.

    Returns dict with keys: baseline_train, baseline_val, labels_train, labels_val,
    perturbed_train, perturbed_val, original_thresholds, perturbation_fractions.
    """
    if cache_dir and not force and _feature_cache_exists(cache_dir):
        return _load_feature_cache(cache_dir)

    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    if isinstance(model, SpikingSequential):
        layer = model.layers[layer_idx]
    else:
        layer = model
    assert isinstance(layer, ConvIntegrateAndFireLayer)

    weights_4d = layer.weights_4d.detach().cpu()
    original_thresholds = layer.thresholds.detach().cpu().clone()
    num_filters = layer.num_filters
    stride = layer.stride
    pad = layer.padding

    perturbation_fractions = PERTURBATION_FRACTIONS
    num_fracs = len(perturbation_fractions)

    # Build threshold matrix: (num_fracs, F)
    thresholds_2d = torch.stack(
        [original_thresholds * (1.0 + frac) for frac in perturbation_fractions]
    )

    # Process each split (train, val)
    results = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        all_times, all_labels = collect_conv_input_times(loader)
        N = all_times.shape[0]

        # Compute flat feature dim analytically
        H, W = all_times.shape[2], all_times.shape[3]
        oH = (H + 2 * pad - layer.kernel_size) // stride + 1
        oW = (W + 2 * pad - layer.kernel_size) // stride + 1
        pool_h = min(pool_size, oH)
        pool_w = min(pool_size, oW)
        flat_dim = num_filters * pool_h * pool_w

        # Compute baseline features in chunks
        baseline_parts = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            st = layer.infer_spike_times_batch(all_times[start:end])
            feat = spike_times_to_features(st, t_target)
            pooled = sum_pool_features(feat, pool_size)
            baseline_parts.append(pooled.flatten(1).numpy())
            del st, feat, pooled
        baseline_features = np.concatenate(baseline_parts, axis=0)

        # Compute perturbed features in chunks using multi-threshold approach
        perturbed_features = np.zeros((num_fracs, N, flat_dim), dtype=np.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_times = all_times[start:end]

            chunk_spike_times = multi_threshold_conv_accumulate(
                chunk_times,
                weights_4d,
                thresholds_2d,
                stride=stride,
                padding=pad,
                device=device,
            )
            chunk_features = _spike_times_to_pooled_features(
                chunk_spike_times,
                t_target,
                pool_size,
            )
            perturbed_features[:, start:end, :] = chunk_features
            del chunk_spike_times, chunk_features

        results[f"baseline_{split_name}"] = baseline_features
        results[f"labels_{split_name}"] = all_labels.numpy()
        results[f"perturbed_{split_name}"] = perturbed_features

    features = {
        **results,
        "original_thresholds": original_thresholds.tolist(),
        "perturbation_fractions": perturbation_fractions,
    }

    if cache_dir:
        _save_feature_cache(cache_dir, features)

    return features


def evaluate_conv_perturbations(
    *,
    features: dict,
    num_filters: int,
    pool_size: int,
    cache_dir: str | None = None,
    classifier_factory: Callable | None = None,
    refit: bool = False,
) -> dict:
    """Per-filter perturbation evaluation for conv layers.

    Like the FC evaluate_perturbations but swaps pool_size*pool_size columns
    per filter instead of 1 column per neuron.

    Args:
        features: Dict from compute_conv_perturbed_features.
        num_filters: Number of conv filters.
        pool_size: Spatial pool size used during feature extraction.
        cache_dir: Optional directory for incremental saves.
        classifier_factory: Callable returning a classifier (default RidgeClassifier).
        refit: If True, refit classifier for each perturbation.

    Returns:
        Dict with baseline metrics, accuracy/f1 matrices, optimal thresholds.
    """
    if classifier_factory is None:
        classifier_factory = RidgeClassifier

    X_train = features["baseline_train"]
    X_val = features["baseline_val"]
    y_train = features["labels_train"]
    y_val = features["labels_val"]
    perturbed_train = features["perturbed_train"]
    perturbed_val = features["perturbed_val"]
    original_thresholds = features["original_thresholds"]
    perturbation_fractions = features["perturbation_fractions"]

    num_fracs = len(perturbation_fractions)
    cols_per_filter = pool_size * pool_size

    # Baseline classifier
    baseline_clf = classifier_factory()
    baseline_clf.fit(X_train, y_train)
    baseline_metrics = compute_metrics(y_val, baseline_clf.predict(X_val))

    # Load partial results if resuming
    accuracy_matrix = np.zeros((num_filters, num_fracs))
    f1_matrix = np.zeros((num_filters, num_fracs))
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

    remaining = [i for i in range(num_fracs) if i not in completed_fractions]
    pbar = tqdm(
        remaining,
        desc="Conv perturbation eval",
        initial=len(completed_fractions),
        total=num_fracs,
    )
    for frac_idx in pbar:
        frac = perturbation_fractions[frac_idx]
        pbar.set_postfix_str(f"frac={frac:+.3f} ({num_filters} filters)")

        for filter_idx in range(num_filters):
            col_start = filter_idx * cols_per_filter
            col_end = col_start + cols_per_filter

            X_train_mod = X_train.copy()
            X_train_mod[:, col_start:col_end] = perturbed_train[
                frac_idx, :, col_start:col_end
            ]
            X_val_mod = X_val.copy()
            X_val_mod[:, col_start:col_end] = perturbed_val[
                frac_idx, :, col_start:col_end
            ]

            if refit:
                clf = classifier_factory()
                clf.fit(X_train_mod, y_train)
                y_pred = clf.predict(X_val_mod)
            else:
                y_pred = baseline_clf.predict(X_val_mod)
            metrics = compute_metrics(y_val, y_pred)
            accuracy_matrix[filter_idx, frac_idx] = metrics["accuracy"]
            f1_matrix[filter_idx, frac_idx] = metrics["f1"]

        completed_fractions.add(frac_idx)
        if cache_dir:
            _save_partial_results(
                cache_dir, completed_fractions, accuracy_matrix, f1_matrix
            )

    # Find per-filter optimal perturbation
    best_frac_indices = accuracy_matrix.argmax(axis=1)
    optimal_fracs = [perturbation_fractions[i] for i in best_frac_indices]
    optimal_thresholds = [
        original_thresholds[n] * (1.0 + optimal_fracs[n]) for n in range(num_filters)
    ]
    optimal_deltas = [
        optimal_thresholds[n] - original_thresholds[n] for n in range(num_filters)
    ]

    return {
        "baseline": baseline_metrics,
        "original_thresholds": original_thresholds,
        "perturbation_fractions": perturbation_fractions,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "f1_matrix": f1_matrix.tolist(),
        "optimal_thresholds": optimal_thresholds,
        "optimal_deltas": optimal_deltas,
        **(
            {
                "baseline_classifier_coef": baseline_clf.coef_.tolist(),
                "baseline_classifier_intercept": baseline_clf.intercept_.tolist(),
            }
            if hasattr(baseline_clf, "coef_")
            else {}
        ),
    }


def run_conv_perturbation_sweep(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    layer_idx: int = 0,
    t_target: float | None = None,
    pool_size: int = 2,
    seed: int = 42,
    cache_dir: str | None = None,
    force: bool = False,
    device: str = "cpu",
    chunk_size: int = 64,
) -> dict:
    """Run per-filter threshold perturbation sweep for a conv model.

    For each filter, perturbs its threshold by fractions from -0.5 to +0.25,
    recomputes that filter's pooled features, and measures accuracy impact.
    """
    features = compute_conv_perturbed_features(
        model_path=model_path,
        dataset_loaders=dataset_loaders,
        layer_idx=layer_idx,
        t_target=t_target,
        pool_size=pool_size,
        seed=seed,
        cache_dir=cache_dir,
        force=force,
        device=device,
        chunk_size=chunk_size,
    )
    num_filters = len(features["original_thresholds"])
    return evaluate_conv_perturbations(
        features=features,
        num_filters=num_filters,
        pool_size=pool_size,
        cache_dir=cache_dir,
    )
