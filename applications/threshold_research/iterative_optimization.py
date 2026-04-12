"""Iterative coordinate descent threshold optimization for conv SNN layers.

Each round: recompute features at current thresholds ±step, refit Ridge,
run sequential greedy with Woodbury updates. Repeat until convergence.

Baseline features are computed on GPU only for round 1; subsequent rounds
reuse the working feature matrices from the previous round (filters are
independent during inference, so per-filter column swaps compose correctly).
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from applications.threshold_research.conv_neuron_perturbation import (
    collect_conv_input_times,
    multi_threshold_conv_accumulate,
    _spike_times_to_pooled_features,
)
from applications.threshold_research.filter_ordering import ORDERINGS, get_filter_order
from spiking import load_model
from spiking.evaluation.column_swap_classifier import ColumnSwapClassifier
from spiking.evaluation.eval_utils import compute_metrics
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap
from spiking.evaluation.svc_column_swap import SVCColumnSwap
from spiking.layers import SpikingSequential
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer

logger = logging.getLogger(__name__)

CLASSIFIERS = ["ridge", "svc"]


def _gpu_available(backend: str) -> bool:
    """Check if GPU backend is available (cupy for ridge, cuml for svc)."""
    try:
        if backend == "ridge":
            import cupy  # noqa: F401

            return True
        if backend == "svc":
            import cuml  # noqa: F401

            return True
    except ImportError:
        pass
    return False


def _make_classifier(name: str, alpha: float = 1.0) -> ColumnSwapClassifier:
    """Create a ColumnSwapClassifier, using GPU automatically if available."""
    use_gpu = _gpu_available(name)
    if name == "ridge":
        return RidgeColumnSwap(alpha=alpha, use_gpu=use_gpu)
    if name == "svc":
        return SVCColumnSwap(use_gpu=use_gpu)
    raise ValueError(f"Unknown classifier: {name!r}. Choose from {CLASSIFIERS}")


def _extract_perturbations(
    all_times: torch.Tensor,
    weights_4d: torch.Tensor,
    current_thresholds: torch.Tensor,
    step_size: float,
    min_threshold: float,
    stride: int,
    padding: int,
    t_target: float | None,
    pool_size: int,
    device: str,
    chunk_size: int,
    desc_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract minus and plus perturbation features in a single GPU pass.

    :returns: (minus_features, plus_features) each (N, flat_dim).
    """
    N = all_times.shape[0]
    minus_thresh = (current_thresholds - step_size).clamp(min=min_threshold)
    plus_thresh = current_thresholds + step_size
    thresholds_2d = torch.stack([minus_thresh, plus_thresh])  # (2, F)

    n_chunks = (N + chunk_size - 1) // chunk_size
    parts = []
    for start in tqdm(
        range(0, N, chunk_size),
        desc=desc_prefix,
        total=n_chunks,
        leave=False,
    ):
        end = min(start + chunk_size, N)
        st = multi_threshold_conv_accumulate(
            all_times[start:end],
            weights_4d,
            thresholds_2d,
            stride=stride,
            padding=padding,
            device=device,
        )
        feat = _spike_times_to_pooled_features(st, t_target, pool_size)
        parts.append(feat)
        del st
    # (2, N, flat_dim)
    all_feat = np.concatenate(parts, axis=1)
    return all_feat[0], all_feat[1]


def _extract_baseline(
    all_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds: torch.Tensor,
    stride: int,
    padding: int,
    t_target: float | None,
    pool_size: int,
    device: str,
    chunk_size: int,
    desc_prefix: str = "",
) -> np.ndarray:
    """Extract features for a single threshold vector."""
    N = all_times.shape[0]
    thresholds_2d = thresholds.unsqueeze(0)  # (1, F)

    n_chunks = (N + chunk_size - 1) // chunk_size
    parts = []
    for start in tqdm(
        range(0, N, chunk_size),
        desc=desc_prefix,
        total=n_chunks,
        leave=False,
    ):
        end = min(start + chunk_size, N)
        st = multi_threshold_conv_accumulate(
            all_times[start:end],
            weights_4d,
            thresholds_2d,
            stride=stride,
            padding=padding,
            device=device,
        )
        feat = _spike_times_to_pooled_features(st, t_target, pool_size)
        parts.append(feat[0])
        del st
    return np.concatenate(parts, axis=0)


def iterative_descent_round(
    baseline_train: np.ndarray,
    baseline_val: np.ndarray,
    minus_train: np.ndarray,
    minus_val: np.ndarray,
    plus_train: np.ndarray,
    plus_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    current_thresholds: list[float],
    step_size: float,
    min_threshold: float,
    num_filters: int,
    pool_size: int,
    alpha: float = 1.0,
    ordering: str = "descending_importance",
    training_spike_times: np.ndarray | None = None,
    classifier: str = "ridge",
) -> tuple[list[float], dict, np.ndarray, np.ndarray]:
    """Run one round of coordinate descent over all filters.

    :param classifier: "ridge" (Woodbury) or "svc" (refit). GPU used automatically.
    :returns: (updated_thresholds, round_info, final_X_train, final_X_val).
    """
    cols_per_filter = pool_size * pool_size
    X_train = baseline_train.copy()
    X_val = baseline_val.copy()

    clf = _make_classifier(classifier, alpha=alpha)
    clf.fit(X_train, y_train)

    baseline_train_acc = compute_metrics(y_train, clf.predict(X_train))["accuracy"]
    baseline_val_acc = compute_metrics(y_val, clf.predict(X_val))["accuracy"]

    # Rank filters by importance and mean spike time
    importance = np.mean(np.abs(clf.weights), axis=1)
    filter_importance = np.array(
        [
            importance[f * cols_per_filter : (f + 1) * cols_per_filter].sum()
            for f in range(num_filters)
        ]
    )
    mean_spike_times = np.array(
        [
            X_train[:, f * cols_per_filter : (f + 1) * cols_per_filter].mean()
            for f in range(num_filters)
        ]
    )
    thresholds_arr = np.array(current_thresholds)
    threshold_deviation = np.abs(thresholds_arr - thresholds_arr.mean())
    filter_order = get_filter_order(
        ordering,
        filter_importance,
        mean_spike_times,
        threshold_drift=threshold_deviation,
        training_spike_times=training_spike_times,
    )

    updated_thresholds = list(current_thresholds)
    current_train_acc = baseline_train_acc
    num_changes = 0
    total_abs_change = 0.0

    # Options: 0=minus, 1=baseline(no change), 2=plus
    option_features_train = [minus_train, baseline_train, plus_train]
    option_features_val = [minus_val, baseline_val, plus_val]
    option_deltas = [-step_size, 0.0, step_size]

    for filter_idx in filter_order:
        filter_idx = int(filter_idx)
        col_start = filter_idx * cols_per_filter
        col_end = col_start + cols_per_filter
        col_indices = list(range(col_start, col_end))

        best_option = 1  # baseline = no change
        best_acc = current_train_acc

        for opt_idx in [0, 2]:  # only test minus and plus
            new_threshold = updated_thresholds[filter_idx] + option_deltas[opt_idx]
            if new_threshold < min_threshold:
                continue

            new_train_cols = option_features_train[opt_idx][:, col_start:col_end]
            X_train_mod = X_train.copy()
            X_train_mod[:, col_start:col_end] = new_train_cols

            y_pred = clf.predict_swapped(col_indices, new_train_cols, X_train_mod)
            acc = compute_metrics(y_train, y_pred)["accuracy"]

            if acc > best_acc:
                best_acc = acc
                best_option = opt_idx

        if best_option != 1:
            delta = option_deltas[best_option]
            best_train_cols = option_features_train[best_option][:, col_start:col_end]
            clf.apply_swap(col_indices, best_train_cols)

            X_train[:, col_start:col_end] = best_train_cols
            X_val[:, col_start:col_end] = option_features_val[best_option][
                :, col_start:col_end
            ]

            updated_thresholds[filter_idx] += delta
            current_train_acc = best_acc
            num_changes += 1
            total_abs_change += abs(delta)

    final_train_acc = compute_metrics(y_train, clf.predict(X_train))["accuracy"]
    final_val_acc = compute_metrics(y_val, clf.predict(X_val))["accuracy"]

    round_info = {
        "num_changes": num_changes,
        "total_abs_change": total_abs_change,
        "baseline_train_accuracy": baseline_train_acc,
        "baseline_val_accuracy": baseline_val_acc,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
    }
    return updated_thresholds, round_info, X_train, X_val


def iterative_coordinate_descent(
    *,
    model_path: str,
    dataset_loaders: tuple,
    t_target: float | None = None,
    pool_size: int = 2,
    min_threshold: float = 1.0,
    device: str = "cpu",
    chunk_size: int = 128,
    num_rounds: int = 25,
    step_size: float = 0.2,
    alpha: float = 1.0,
    ordering: str = "descending_importance",
    classifier: str = "ridge",
) -> dict:
    """Run iterative coordinate descent threshold optimization."""
    model = load_model(model_path)
    if isinstance(model, SpikingSequential):
        layer = model.layers[0]
    else:
        layer = model
    assert isinstance(layer, ConvIntegrateAndFireLayer)

    weights_4d = layer.weights_4d.detach().cpu()
    original_thresholds = layer.thresholds.detach().cpu().clone()
    num_filters = layer.num_filters
    stride = layer.stride
    padding = layer.padding

    # Collect input times once (reused across rounds)
    train_loader, val_loader = dataset_loaders
    logger.info("Collecting input times...")
    train_times, train_labels = collect_conv_input_times(train_loader, chunk_size)
    val_times, val_labels = collect_conv_input_times(val_loader, chunk_size)
    y_train = train_labels.numpy()
    y_val = val_labels.numpy()

    current_thresholds = original_thresholds.clone()
    rounds_history = []
    cumulative_abs_change = 0.0

    # Load training spike times from log if available (for training_*_spike orderings)
    training_spike_times: np.ndarray | None = None
    training_metrics_path = os.path.join(
        os.path.dirname(model_path), "training_metrics.json"
    )
    if os.path.exists(training_metrics_path):
        with open(training_metrics_path) as f:
            _tm = json.load(f)
        raw = _tm.get("mean_spike_time_per_neuron")
        if raw is not None:
            training_spike_times = np.array(
                [float("nan") if v is None else v for v in raw]
            )

    # Shared kwargs for GPU feature extraction
    gpu_kwargs = dict(
        weights_4d=weights_4d,
        stride=stride,
        padding=padding,
        t_target=t_target,
        pool_size=pool_size,
        device=device,
        chunk_size=chunk_size,
    )

    # Round 1: compute baseline on GPU
    logger.info("Round 1/%d — extracting baseline + perturbations...", num_rounds)
    base_train = _extract_baseline(
        train_times,
        thresholds=current_thresholds,
        desc_prefix="  R1 baseline train",
        **gpu_kwargs,
    )
    base_val = _extract_baseline(
        val_times,
        thresholds=current_thresholds,
        desc_prefix="  R1 baseline val",
        **gpu_kwargs,
    )

    for round_idx in range(num_rounds):
        logger.info(
            "Round %d/%d — extracting perturbations...", round_idx + 1, num_rounds
        )

        # GPU: extract only ±step features (baseline reused from previous round)
        minus_train, plus_train = _extract_perturbations(
            train_times,
            current_thresholds=current_thresholds,
            step_size=step_size,
            min_threshold=min_threshold,
            desc_prefix=f"  R{round_idx + 1} train",
            **gpu_kwargs,
        )
        minus_val, plus_val = _extract_perturbations(
            val_times,
            current_thresholds=current_thresholds,
            step_size=step_size,
            min_threshold=min_threshold,
            desc_prefix=f"  R{round_idx + 1} val",
            **gpu_kwargs,
        )

        # CPU: sequential greedy with Woodbury
        logger.info("Round %d/%d — optimizing...", round_idx + 1, num_rounds)
        new_thresholds, round_info, base_train, base_val = iterative_descent_round(
            baseline_train=base_train,
            baseline_val=base_val,
            minus_train=minus_train,
            minus_val=minus_val,
            plus_train=plus_train,
            plus_val=plus_val,
            y_train=y_train,
            y_val=y_val,
            current_thresholds=current_thresholds.tolist(),
            step_size=step_size,
            min_threshold=min_threshold,
            num_filters=num_filters,
            pool_size=pool_size,
            alpha=alpha,
            ordering=ordering,
            training_spike_times=training_spike_times,
            classifier=classifier,
        )

        cumulative_abs_change += round_info["total_abs_change"]
        round_info["round"] = round_idx
        round_info["cumulative_abs_change"] = cumulative_abs_change
        rounds_history.append(round_info)

        current_thresholds = torch.tensor(
            new_thresholds, dtype=original_thresholds.dtype
        )

        logger.info(
            "  Round %d: %d changes, abs_change=%.1f, train=%.4f->%.4f, val=%.4f->%.4f",
            round_idx + 1,
            round_info["num_changes"],
            round_info["total_abs_change"],
            round_info["baseline_train_accuracy"],
            round_info["final_train_accuracy"],
            round_info["baseline_val_accuracy"],
            round_info["final_val_accuracy"],
        )

        if round_info["num_changes"] == 0:
            logger.info("  Converged at round %d (no changes).", round_idx + 1)
            break

    return {
        "config": {
            "num_rounds": num_rounds,
            "step_size": step_size,
            "alpha": alpha,
            "ordering": ordering,
            "classifier": classifier,
        },
        "baseline_train_accuracy": rounds_history[0]["baseline_train_accuracy"],
        "baseline_val_accuracy": rounds_history[0]["baseline_val_accuracy"],
        "final_train_accuracy": rounds_history[-1]["final_train_accuracy"],
        "final_val_accuracy": rounds_history[-1]["final_val_accuracy"],
        "original_thresholds": original_thresholds.tolist(),
        "optimized_thresholds": current_thresholds.tolist(),
        "rounds": rounds_history,
    }


def plot_convergence(results: dict, output_path: str) -> None:
    """Plot convergence: cumulative threshold change and accuracy over rounds."""
    rounds = results["rounds"]
    xs = [r["round"] + 1 for r in rounds]
    cum_change = [r["cumulative_abs_change"] for r in rounds]
    train_acc = [r["final_train_accuracy"] for r in rounds]
    val_acc = [r["final_val_accuracy"] for r in rounds]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_change = "tab:blue"
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative absolute threshold change", color=color_change)
    ax1.plot(xs, cum_change, "o-", color=color_change, label="Threshold change")
    ax1.tick_params(axis="y", labelcolor=color_change)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(xs, train_acc, "s--", color="tab:orange", label="Train acc", alpha=0.8)
    ax2.plot(xs, val_acc, "^--", color="tab:green", label="Val acc", alpha=0.8)
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Iterative Coordinate Descent Convergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved convergence plot to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Iterative coordinate descent threshold optimization"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing model.pth and setup.json",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--num-rounds", type=int, default=25)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        default="ridge",
        help="Classifier for optimization (default: ridge). 'svc' uses LinearSVC.",
    )
    parser.add_argument(
        "--orderings",
        nargs="+",
        choices=ORDERINGS,
        default=None,
        help="Orderings to run (default: all). Choices: %(choices)s",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    active_orderings = args.orderings or ORDERINGS

    with open(f"{model_dir}/setup.json") as f:
        setup = json.load(f)

    dataset = setup["dataset"]
    t_target = setup.get("target_timestamp")
    pool_size = setup.get("pool_size", 2)
    min_threshold = setup.get("min_threshold", 1.0)

    # Load dataset
    if dataset == "cifar10":
        from applications.datasets import Cifar10WhitenedDataset
        from torch.utils.data import DataLoader

        train_ds = Cifar10WhitenedDataset("data", "train")
        val_ds = Cifar10WhitenedDataset(
            "data",
            "test",
            kernels=train_ds.kernels,
            mean=train_ds.mean,
        )
        train_loader = DataLoader(train_ds, batch_size=None, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)
    else:
        from applications.datasets import create_dataset

        train_loader, val_loader = create_dataset(dataset)

    for ordering in active_orderings:
        output_path = f"{model_dir}/iterative_optimization_{ordering}.json"
        plot_path = f"{model_dir}/iterative_optimization_{ordering}_convergence.png"

        if not args.force and os.path.exists(output_path):
            logger.info("Already exists: %s (use --force to re-run)", output_path)
            continue

        results = iterative_coordinate_descent(
            model_path=f"{model_dir}/model.pth",
            dataset_loaders=(train_loader, val_loader),
            t_target=t_target,
            pool_size=pool_size,
            min_threshold=min_threshold,
            device=args.device,
            chunk_size=args.chunk_size,
            num_rounds=args.num_rounds,
            step_size=args.step_size,
            ordering=ordering,
            classifier=args.classifier,
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved results to %s", output_path)

        plot_convergence(results, plot_path)

        logger.info(
            "Summary [%s]: train %.4f -> %.4f (%+.4f), val %.4f -> %.4f (%+.4f), %d rounds",
            ordering,
            results["baseline_train_accuracy"],
            results["final_train_accuracy"],
            results["final_train_accuracy"] - results["baseline_train_accuracy"],
            results["baseline_val_accuracy"],
            results["final_val_accuracy"],
            results["final_val_accuracy"] - results["baseline_val_accuracy"],
            len(results["rounds"]),
        )
