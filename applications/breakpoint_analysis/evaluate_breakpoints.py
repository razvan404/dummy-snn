"""Evaluate accuracy at breakpoint-derived threshold candidates.

Instead of the fixed 25-fraction grid, this uses actual breakpoints in the
[0.75θ, 1.25θ] range — the exact threshold values where spike times change.
For each filter, we sample N breakpoints as evenly-spaced quantiles, then
evaluate accuracy via Woodbury column swap.

This gives a much higher-resolution view of the per-filter accuracy landscape
in the region that matters most.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spiking import load_model
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.eval_utils import compute_metrics
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap
from spiking.layers import SpikingSequential
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer

from applications.breakpoint_analysis.compute_breakpoints import (
    collect_breakpoints_per_filter_chunk_fast,
)
from applications.threshold_research.conv_neuron_perturbation import (
    collect_conv_input_times,
    multi_threshold_conv_accumulate,
)

logger = logging.getLogger(__name__)


def sample_breakpoint_candidates(
    breakpoints: np.ndarray,
    threshold: float,
    num_candidates: int = 50,
    low_ratio: float = 0.75,
    high_ratio: float = 1.25,
) -> np.ndarray:
    """Sample evenly-spaced quantiles from breakpoints near the threshold.

    :param breakpoints: Sorted array of unique cumulative potential values.
    :param threshold: Current threshold value for this filter.
    :param num_candidates: Number of candidates to sample.
    :param low_ratio: Lower bound as fraction of threshold.
    :param high_ratio: Upper bound as fraction of threshold.
    :returns: Array of candidate threshold values (sorted, unique).
    """
    low = threshold * low_ratio
    high = threshold * high_ratio
    mask = (breakpoints >= low) & (breakpoints <= high)
    near = breakpoints[mask]

    if len(near) == 0:
        return np.array([threshold])

    if len(near) <= num_candidates:
        return near

    # Sample evenly-spaced quantiles for uniform coverage
    quantile_indices = np.linspace(0, len(near) - 1, num_candidates, dtype=int)
    return np.unique(near[quantile_indices])


def compute_breakpoint_candidates(
    model_path: str,
    loader: DataLoader,
    num_candidates: int = 50,
    low_ratio: float = 0.75,
    high_ratio: float = 1.25,
    layer_idx: int = 0,
    device: str = "cpu",
    chunk_size: int = 256,
    max_samples: int | None = None,
    breakpoints_path: str | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compute or load breakpoints, then sample candidates per filter.

    :param model_path: Path to saved model.
    :param loader: DataLoader yielding (spike_times, labels).
    :param num_candidates: Candidates per filter.
    :param low_ratio: Lower bound ratio.
    :param high_ratio: Upper bound ratio.
    :param layer_idx: Layer index for sequential models.
    :param device: Computation device.
    :param chunk_size: Batch size.
    :param max_samples: Cap on samples for breakpoint collection.
    :param breakpoints_path: Path to precomputed breakpoints.npz.
    :returns: (candidates_per_filter, thresholds) where candidates_per_filter
        is a list of F arrays and thresholds is (F,) array.
    """
    model = load_model(model_path)
    if isinstance(model, SpikingSequential):
        layer = model.layers[layer_idx]
    else:
        layer = model
    assert isinstance(layer, ConvIntegrateAndFireLayer)

    thresholds = layer.thresholds.detach().cpu().numpy()
    num_filters = layer.num_filters

    # Load or compute breakpoints
    if breakpoints_path and Path(breakpoints_path).exists():
        logger.info("Loading breakpoints from %s", breakpoints_path)
        data = np.load(breakpoints_path)
        per_filter_bps = [data[f"filter_{f}"] for f in range(num_filters)]
    else:
        logger.info("Computing breakpoints from scratch...")
        weights_4d = layer.weights_4d.detach().cpu()
        stride, padding = layer.stride, layer.padding

        batched = DataLoader(loader.dataset, batch_size=chunk_size, shuffle=False)
        all_times_parts = []
        total = 0
        for batch_times, _ in batched:
            all_times_parts.append(batch_times)
            total += batch_times.shape[0]
            if max_samples and total >= max_samples:
                break
        all_times = torch.cat(all_times_parts, dim=0)
        if max_samples:
            all_times = all_times[:max_samples]

        per_filter_bps = [np.array([], dtype=np.float32)] * num_filters
        per_filter_lists: list[list[np.ndarray]] = [[] for _ in range(num_filters)]
        N = all_times.shape[0]
        for start in tqdm(
            range(0, N, chunk_size),
            total=(N + chunk_size - 1) // chunk_size,
            desc="Computing breakpoints",
        ):
            end = min(start + chunk_size, N)
            chunk_bps = collect_breakpoints_per_filter_chunk_fast(
                all_times[start:end], weights_4d, stride, padding, device=device
            )
            for f in range(num_filters):
                if len(chunk_bps[f]) > 0:
                    per_filter_lists[f].append(chunk_bps[f])
            if device != "cpu":
                torch.cuda.empty_cache()

        for f in range(num_filters):
            if per_filter_lists[f]:
                per_filter_bps[f] = np.unique(np.concatenate(per_filter_lists[f]))

    # Sample candidates per filter
    candidates = []
    for f in range(num_filters):
        cands = sample_breakpoint_candidates(
            per_filter_bps[f], thresholds[f], num_candidates, low_ratio, high_ratio
        )
        candidates.append(cands)
        logger.debug(
            "Filter %d: %d candidates in [%.2f, %.2f] (from %d breakpoints)",
            f,
            len(cands),
            thresholds[f] * low_ratio,
            thresholds[f] * high_ratio,
            len(per_filter_bps[f]),
        )

    return candidates, thresholds


def evaluate_breakpoint_candidates(
    model_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    candidates_per_filter: list[np.ndarray],
    layer_idx: int = 0,
    t_target: float | None = None,
    pool_size: int = 2,
    device: str = "cpu",
    chunk_size: int = 256,
    alpha: float = 1.0,
) -> dict:
    """Evaluate accuracy at breakpoint-derived candidates for each filter.

    For each filter, tests each candidate threshold using Woodbury column swap.
    All other filters stay at baseline thresholds.

    :returns: Dict with per-filter results: candidates, accuracies, best threshold.
    """
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
    padding = layer.padding

    # Collect all input times
    train_times, train_labels = collect_conv_input_times(train_loader, chunk_size)
    val_times, val_labels = collect_conv_input_times(val_loader, chunk_size)

    # Compute output spatial size
    H, W = train_times.shape[2], train_times.shape[3]
    oH = (H + 2 * padding - layer.kernel_size) // stride + 1
    oW = (W + 2 * padding - layer.kernel_size) // stride + 1
    pool_h = min(pool_size, oH)
    pool_w = min(pool_size, oW)
    cols_per_filter = pool_h * pool_w

    # Compute baseline features
    logger.info("Computing baseline features...")
    baseline_train = _compute_features_chunked(
        train_times, layer, t_target, pool_size, device, chunk_size, "Baseline train"
    )
    baseline_val = _compute_features_chunked(
        val_times, layer, t_target, pool_size, device, chunk_size, "Baseline val"
    )

    # Fit baseline classifier
    clf = RidgeColumnSwap(alpha=alpha)
    clf.fit(baseline_train, train_labels.numpy())
    baseline_preds = clf.predict(baseline_val)
    baseline_metrics = compute_metrics(val_labels.numpy(), baseline_preds)
    logger.info("Baseline accuracy: %.4f", baseline_metrics["accuracy"])

    # Evaluate each filter's candidates
    filter_results = []
    for f in tqdm(range(num_filters), desc="Evaluating filters"):
        candidates = candidates_per_filter[f]
        num_cands = len(candidates)

        if num_cands == 0:
            filter_results.append(
                {
                    "filter_idx": f,
                    "current_threshold": float(original_thresholds[f]),
                    "candidates": [],
                    "train_accuracies": [],
                    "val_accuracies": [],
                    "best_threshold": float(original_thresholds[f]),
                    "best_val_accuracy": baseline_metrics["accuracy"],
                }
            )
            continue

        # Build threshold matrix: (num_cands, F) — only filter f varies
        thresholds_2d = original_thresholds.unsqueeze(0).repeat(num_cands, 1)
        thresholds_2d[:, f] = torch.from_numpy(candidates).float()

        # Compute perturbed features for train and val
        col_start = f * cols_per_filter
        col_end = col_start + cols_per_filter
        col_indices = list(range(col_start, col_end))

        train_features_perturbed = _compute_perturbed_features_chunked(
            train_times,
            weights_4d,
            thresholds_2d,
            stride,
            padding,
            t_target,
            pool_size,
            device,
            chunk_size,
        )
        val_features_perturbed = _compute_perturbed_features_chunked(
            val_times,
            weights_4d,
            thresholds_2d,
            stride,
            padding,
            t_target,
            pool_size,
            device,
            chunk_size,
        )

        # Evaluate each candidate via Woodbury
        train_accs = []
        val_accs = []
        for c_idx in range(num_cands):
            new_train_cols = train_features_perturbed[c_idx, :, col_start:col_end]
            X_val_mod = baseline_val.copy()
            X_val_mod[:, col_start:col_end] = val_features_perturbed[
                c_idx, :, col_start:col_end
            ]

            y_pred_val = clf.predict_swapped(col_indices, new_train_cols, X_val_mod)
            val_metrics = compute_metrics(val_labels.numpy(), y_pred_val)
            val_accs.append(val_metrics["accuracy"])

            # Also check train accuracy
            X_train_mod = baseline_train.copy()
            X_train_mod[:, col_start:col_end] = new_train_cols
            y_pred_train = clf.predict_swapped(col_indices, new_train_cols, X_train_mod)
            train_metrics = compute_metrics(train_labels.numpy(), y_pred_train)
            train_accs.append(train_metrics["accuracy"])

        best_idx = int(np.argmax(val_accs))
        filter_results.append(
            {
                "filter_idx": f,
                "current_threshold": float(original_thresholds[f]),
                "candidates": candidates.tolist(),
                "train_accuracies": train_accs,
                "val_accuracies": val_accs,
                "best_threshold": float(candidates[best_idx]),
                "best_val_accuracy": float(val_accs[best_idx]),
                "best_train_accuracy": float(train_accs[best_idx]),
            }
        )

    return {
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_f1": baseline_metrics["f1"],
        "num_filters": num_filters,
        "filters": filter_results,
    }


def _compute_features_chunked(
    all_times: torch.Tensor,
    layer: ConvIntegrateAndFireLayer,
    t_target: float | None,
    pool_size: int,
    device: str,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    """Compute baseline features in chunks."""
    N = all_times.shape[0]
    layer.to(device)
    layer.eval()
    parts = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        st = layer.infer_spike_times_batch(all_times[start:end].to(device))
        feat = spike_times_to_features(st.cpu(), t_target)
        pooled = sum_pool_features(feat, pool_size)
        parts.append(pooled.flatten(1).numpy())
    layer.cpu()
    return np.concatenate(parts, axis=0)


def _compute_perturbed_features_chunked(
    all_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int,
    padding: int,
    t_target: float | None,
    pool_size: int,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    """Compute perturbed features for multiple threshold sets in chunks.

    :returns: (num_cands, N, flat_dim) numpy array.
    """
    N = all_times.shape[0]
    num_cands = thresholds_2d.shape[0]
    num_filters = thresholds_2d.shape[1]

    # Compute flat feature dim
    H, W = all_times.shape[2], all_times.shape[3]
    kH = weights_4d.shape[2]
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kH) // stride + 1
    pool_h = min(pool_size, oH)
    pool_w = min(pool_size, oW)
    flat_dim = num_filters * pool_h * pool_w

    result = np.zeros((num_cands, N, flat_dim), dtype=np.float32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_st = multi_threshold_conv_accumulate(
            all_times[start:end],
            weights_4d,
            thresholds_2d,
            stride=stride,
            padding=padding,
            device=device,
        )
        # chunk_st: (num_cands, B_chunk, F, oH, oW)
        nf, B_chunk, F_dim, cH, cW = chunk_st.shape
        flat = chunk_st.reshape(nf * B_chunk, F_dim, cH, cW)
        feat = spike_times_to_features(flat, t_target)
        pooled = sum_pool_features(feat, pool_size)
        flat_feat = pooled.flatten(1).numpy()
        result[:, start:end, :] = flat_feat.reshape(nf, B_chunk, -1)

    return result
