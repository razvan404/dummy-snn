"""Sequential greedy threshold optimization for conv SNN layers.

Precomputes all perturbed features in one GPU pass (filters are independent
during inference), then runs sequential Woodbury evaluation on CPU:
for each filter in importance order, tests all fractions against the
current Ridge state, picks the best, permanently applies via apply_swap.

This is both fast AND accounts for Ridge feature dependencies.
"""

import argparse
import json
import logging
import os

import numpy as np
from tqdm import tqdm

from applications.datasets import DATASETS, create_dataset
from applications.threshold_research.perturbation_params import (
    get_perturbation_params,
)
from applications.threshold_research.conv_neuron_perturbation import (
    compute_conv_perturbed_features,
)
from spiking.evaluation.eval_utils import compute_metrics
from spiking.evaluation.ridge_column_swap import RidgeColumnSwap
from applications.threshold_research.filter_ordering import ORDERINGS, get_filter_order

logger = logging.getLogger(__name__)


def sequential_greedy_from_features(
    features: dict,
    num_filters: int,
    pool_size: int,
    alpha: float = 1.0,
    ordering: str = "descending_importance",
) -> dict:
    """Sequential greedy optimization using precomputed perturbed features.

    For each filter (ordered by classifier importance), tests all perturbation
    fractions against the current Ridge state, picks the best, permanently
    applies the Woodbury update, then moves to the next filter.

    :param features: Dict from compute_conv_perturbed_features with
        baseline_train, baseline_val, perturbed_train, perturbed_val, etc.
    :param num_filters: Number of conv filters.
    :param pool_size: Spatial pool size.
    :param alpha: Ridge regularization strength.
    :param ordering: One of ORDERINGS — controls which filters are optimized first.
    :returns: Dict with optimization history and final thresholds.
    """
    X_train = features["baseline_train"].copy()
    X_val = features["baseline_val"].copy()
    y_train = features["labels_train"]
    y_val = features["labels_val"]
    perturbed_train = features["perturbed_train"]
    perturbed_val = features["perturbed_val"]
    original_thresholds = features["original_thresholds"]
    perturbation_fractions = features["perturbation_fractions"]

    num_fracs = len(perturbation_fractions)
    cols_per_filter = pool_size * pool_size

    # Fit baseline Ridge with Woodbury support
    clf = RidgeColumnSwap(alpha=alpha)
    clf.fit(X_train, y_train)
    baseline_train_metrics = compute_metrics(y_train, clf.predict(X_train))
    baseline_val_metrics = compute_metrics(y_val, clf.predict(X_val))
    logger.info(
        "Baseline — train: %.4f, val: %.4f",
        baseline_train_metrics["accuracy"],
        baseline_val_metrics["accuracy"],
    )

    # Compute per-filter importance (Ridge coefficient magnitude) and mean spike time
    importance = np.mean(np.abs(clf._w), axis=1)  # (d,)
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
    thresholds_arr = np.array(original_thresholds)
    threshold_deviation = np.abs(thresholds_arr - thresholds_arr.mean())
    filter_order = get_filter_order(
        ordering,
        filter_importance,
        mean_spike_times,
        threshold_drift=threshold_deviation,
        training_spike_times=features.get("training_spike_times"),
    )
    logger.info("Filter ordering: %s", ordering)

    # Sequential greedy optimization — select fractions based on TRAIN accuracy
    # to avoid leaking val information into threshold selection.
    history = []
    current_thresholds = list(original_thresholds)
    current_train_accuracy = baseline_train_metrics["accuracy"]

    pbar = tqdm(filter_order, desc="Sequential optimization")
    for filter_idx in pbar:
        filter_idx = int(filter_idx)
        col_start = filter_idx * cols_per_filter
        col_end = col_start + cols_per_filter
        col_indices = list(range(col_start, col_end))

        best_frac_idx = -1
        best_train_accuracy = current_train_accuracy

        for frac_idx in range(num_fracs):
            new_train_cols = perturbed_train[frac_idx, :, col_start:col_end]
            # Evaluate on train to select the best fraction
            X_train_mod = X_train.copy()
            X_train_mod[:, col_start:col_end] = new_train_cols

            y_pred = clf.predict_swapped(col_indices, new_train_cols, X_train_mod)
            acc = compute_metrics(y_train, y_pred)["accuracy"]

            if acc > best_train_accuracy:
                best_train_accuracy = acc
                best_frac_idx = frac_idx

        # Apply the best fraction permanently
        if best_frac_idx >= 0:
            best_frac = perturbation_fractions[best_frac_idx]
            new_threshold = current_thresholds[filter_idx] * (1.0 + best_frac)

            # Permanently apply Woodbury update
            best_train_cols = perturbed_train[best_frac_idx, :, col_start:col_end]
            clf.apply_swap(col_indices, best_train_cols)

            # Update validation features (for final reporting, not selection)
            X_val[:, col_start:col_end] = perturbed_val[
                best_frac_idx, :, col_start:col_end
            ]
            # Update train features working copy
            X_train[:, col_start:col_end] = best_train_cols

            current_thresholds[filter_idx] = new_threshold
            current_train_accuracy = best_train_accuracy

            history.append(
                {
                    "step": len(history),
                    "filter": filter_idx,
                    "fraction": best_frac,
                    "new_threshold": new_threshold,
                    "train_accuracy": best_train_accuracy,
                }
            )
            pbar.set_postfix_str(
                f"f={filter_idx}, frac={best_frac:+.4f}, train={best_train_accuracy:.4f}"
            )
        else:
            history.append(
                {
                    "step": len(history),
                    "filter": filter_idx,
                    "fraction": 0.0,
                    "new_threshold": current_thresholds[filter_idx],
                    "train_accuracy": current_train_accuracy,
                }
            )
            pbar.set_postfix_str(
                f"f={filter_idx}, skip, train={current_train_accuracy:.4f}"
            )

    # Final evaluation on both splits
    final_train_metrics = compute_metrics(y_train, clf.predict(X_train))
    final_val_metrics = compute_metrics(y_val, clf.predict(X_val))
    logger.info(
        "Final — train: %.4f (%+.4f), val: %.4f (%+.4f)",
        final_train_metrics["accuracy"],
        final_train_metrics["accuracy"] - baseline_train_metrics["accuracy"],
        final_val_metrics["accuracy"],
        final_val_metrics["accuracy"] - baseline_val_metrics["accuracy"],
    )

    return {
        "baseline_train": baseline_train_metrics,
        "baseline_val": baseline_val_metrics,
        "final_train": final_train_metrics,
        "final_val": final_val_metrics,
        "train_improvement": final_train_metrics["accuracy"]
        - baseline_train_metrics["accuracy"],
        "val_improvement": final_val_metrics["accuracy"]
        - baseline_val_metrics["accuracy"],
        "original_thresholds": original_thresholds,
        "optimized_thresholds": current_thresholds,
        "ordering": ordering,
        "filter_order": filter_order.tolist(),
        "perturbation_fractions": perturbation_fractions,
        "history": history,
        "num_improved": sum(1 for h in history if h["fraction"] != 0.0),
    }


def run_sequential_optimization(
    *,
    model_path: str,
    dataset_loaders: tuple,
    t_target: float | None = None,
    pool_size: int = 2,
    seed: int = 42,
    cache_dir: str | None = None,
    force: bool = False,
    device: str = "cpu",
    chunk_size: int = 128,
    alpha: float = 1.0,
    ordering: str = "descending_importance",
) -> dict:
    """Precompute features on GPU, then run sequential Woodbury optimization.

    Phase 1 (GPU): compute baseline + all perturbed features in one pass.
    Phase 2 (CPU): sequential greedy Woodbury evaluation.
    """
    # Phase 1: precompute all features (reuse independent sweep's GPU code)
    logger.info("Phase 1: Computing perturbed features (GPU)...")
    features = compute_conv_perturbed_features(
        model_path=model_path,
        dataset_loaders=dataset_loaders,
        t_target=t_target,
        pool_size=pool_size,
        seed=seed,
        cache_dir=cache_dir,
        force=force,
        device=device,
        chunk_size=chunk_size,
    )

    # Attach training spike times if available (needed for training_*_spike orderings)
    training_metrics_path = os.path.join(
        os.path.dirname(model_path), "training_metrics.json"
    )
    if os.path.exists(training_metrics_path):
        with open(training_metrics_path) as f:
            training_metrics = json.load(f)
        raw = training_metrics.get("mean_spike_time_per_neuron")
        if raw is not None:
            features["training_spike_times"] = np.array(
                [float("nan") if v is None else v for v in raw]
            )

    # Phase 2: sequential greedy optimization (CPU, fast)
    num_filters = len(features["original_thresholds"])
    logger.info(
        "Phase 2: Sequential greedy optimization (%d filters, ordering=%s)...",
        num_filters,
        ordering,
    )
    return sequential_greedy_from_features(
        features=features,
        num_filters=num_filters,
        pool_size=pool_size,
        alpha=alpha,
        ordering=ordering,
    )


def _find_models(base_dir: str) -> list[tuple[str, float, int]]:
    """Discover model paths under base_dir."""
    models = []
    if not os.path.isdir(base_dir):
        return models
    for tobj_name in sorted(os.listdir(base_dir)):
        if not tobj_name.startswith("tobj_"):
            continue
        try:
            t_obj = float(tobj_name.split("_", 1)[1])
        except ValueError:
            continue
        tobj_path = os.path.join(base_dir, tobj_name)
        for seed_name in sorted(os.listdir(tobj_path)):
            if not seed_name.startswith("seed_"):
                continue
            try:
                seed = int(seed_name.split("_", 1)[1])
            except ValueError:
                continue
            model_path = os.path.join(tobj_path, seed_name, "model.pth")
            if os.path.exists(model_path):
                models.append((model_path, t_obj, seed))
    return models


def run(
    dataset: str,
    *,
    force: bool = False,
    seeds: list[int] | None = None,
    device: str = "cpu",
    chunk_size: int = 128,
    orderings: list[str] | None = None,
):
    """Run sequential optimization for all models and orderings of a dataset."""
    active_orderings = orderings or ORDERINGS
    train_loader, val_loader = create_dataset(dataset)
    params = get_perturbation_params(dataset)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_models(base_dir)
    if seeds:
        models = [(p, t, s) for p, t, s in models if s in seeds]
    if not models:
        print(f"No models found under {base_dir}")
        return

    for model_path, t_obj, seed in models:
        cache_dir = os.path.join(os.path.dirname(model_path), "perturbation_cache")
        features_computed = False

        for ordering in active_orderings:
            output_path = os.path.join(
                os.path.dirname(model_path),
                f"sequential_optimization_{ordering}.json",
            )
            if not force and os.path.exists(output_path):
                print(
                    f"  skip t_obj={t_obj} seed={seed} ordering={ordering} (already complete)"
                )
                continue

            print(f"  running t_obj={t_obj} seed={seed} ordering={ordering}")
            result = run_sequential_optimization(
                model_path=model_path,
                dataset_loaders=(train_loader, val_loader),
                t_target=t_obj,
                pool_size=params["pool_size"],
                seed=seed,
                cache_dir=cache_dir,
                force=force and not features_computed,
                device=device,
                chunk_size=chunk_size,
                ordering=ordering,
            )
            features_computed = True

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

            print(
                f"    train: {result['baseline_train']['accuracy']:.4f} -> {result['final_train']['accuracy']:.4f} ({result['train_improvement']:+.4f})  "
                f"val: {result['baseline_val']['accuracy']:.4f} -> {result['final_val']['accuracy']:.4f} ({result['val_improvement']:+.4f})  "
                f"({result['num_improved']}/{len(result['filter_order'])} filters improved)"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Sequential greedy threshold optimization for conv SNN"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--orderings",
        nargs="+",
        choices=ORDERINGS,
        default=None,
        help="Orderings to run (default: all). Choices: %(choices)s",
    )
    args = parser.parse_args()
    run(
        args.dataset,
        force=args.force,
        seeds=args.seeds,
        device=args.device,
        chunk_size=args.chunk_size,
        orderings=args.orderings,
    )
