import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

from applications.common import set_seed
from applications.datasets import DATASETS, create_dataset
from applications.threshold_research.neuron_perturbation import (
    collect_input_times,
    compute_features_with_thresholds,
    precompute_cumulative_potentials,
)
from applications.threshold_research.threshold_sync import compute_fisher_thresholds
from applications.threshold_research.run_perturbation import _find_models
from spiking import load_model
from spiking.evaluation.eval_utils import compute_metrics
from spiking.layers import SpikingSequential


def _collect_labels(loader: torch.utils.data.DataLoader) -> np.ndarray:
    """Collect all labels from a DataLoader into a single numpy array."""
    batched = torch.utils.data.DataLoader(loader.dataset, batch_size=256, shuffle=False)
    parts = []
    for _times, labels in batched:
        parts.append(labels)
    return torch.cat(parts, dim=0).numpy()


def _precompute_model_data(
    model_path,
    dataset_loaders,
    layer_idx,
    t_target,
    seed,
    status_fn=None,
):
    """Load model and precompute potentials, baseline features, and classifier.

    Returns a dict with all data needed to evaluate Fisher threshold optimization.
    """

    def _status(msg):
        if status_fn:
            status_fn(msg)

    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    _status("loading model")
    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    sub_model = sub_model.cpu()

    original_thresholds = layer.thresholds.detach().clone()
    weights = layer.weights.detach()

    _status("precomputing train potentials")
    train_input_times = collect_input_times(train_loader)
    train_cum, train_boundary = precompute_cumulative_potentials(
        train_input_times, weights
    )

    _status("precomputing val potentials")
    val_input_times = collect_input_times(val_loader)
    val_cum, val_boundary = precompute_cumulative_potentials(val_input_times, weights)

    _status("computing baseline features")
    train_features = compute_features_with_thresholds(
        train_cum, train_boundary, original_thresholds, t_target
    )
    val_features = compute_features_with_thresholds(
        val_cum, val_boundary, original_thresholds, t_target
    )

    train_labels = _collect_labels(train_loader)
    val_labels = _collect_labels(val_loader)

    _status("fitting baseline classifier")
    baseline_clf = LinearSVC(dual=False, tol=1e-3, max_iter=10000)
    baseline_clf.fit(train_features, train_labels)
    baseline_metrics = compute_metrics(val_labels, baseline_clf.predict(val_features))

    return {
        "original_thresholds": original_thresholds,
        "train_cum": train_cum,
        "train_boundary": train_boundary,
        "val_cum": val_cum,
        "val_boundary": val_boundary,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "baseline_clf": baseline_clf,
        "baseline_metrics": baseline_metrics,
        "t_target": t_target,
    }


def _evaluate_fisher(precomputed: dict) -> dict:
    """Evaluate Fisher discriminant threshold optimization."""
    p = precomputed
    fisher_thresholds = compute_fisher_thresholds(
        p["train_cum"],
        p["train_boundary"],
        p["original_thresholds"],
        p["train_labels"],
        t_target=p["t_target"],
    )

    # Count adjusted neurons
    adjusted_mask = fisher_thresholds != p["original_thresholds"]
    n_adjusted = int(adjusted_mask.sum().item())
    threshold_shifts = (fisher_thresholds - p["original_thresholds"])[adjusted_mask]

    # Compute features with Fisher thresholds
    fisher_train_features = compute_features_with_thresholds(
        p["train_cum"], p["train_boundary"], fisher_thresholds, p["t_target"]
    )
    fisher_val_features = compute_features_with_thresholds(
        p["val_cum"], p["val_boundary"], fisher_thresholds, p["t_target"]
    )

    clf = LinearSVC(dual=False, tol=1e-3, max_iter=10000)
    clf.fit(fisher_train_features, p["train_labels"])
    metrics = compute_metrics(p["val_labels"], clf.predict(fisher_val_features))

    shift_stats = {}
    if n_adjusted > 0:
        shift_stats = {
            "mean_shift": float(threshold_shifts.mean()),
            "std_shift": float(threshold_shifts.std()) if n_adjusted > 1 else 0.0,
            "min_shift": float(threshold_shifts.min()),
            "max_shift": float(threshold_shifts.max()),
        }

    return {
        "n_adjusted": n_adjusted,
        **metrics,
        **shift_stats,
    }


def evaluate_fisher_thresholds(
    *,
    model_path,
    dataset_loaders,
    spike_shape,
    layer_idx=0,
    t_target=None,
    seed=42,
):
    """Evaluate Fisher discriminant threshold optimization.

    Returns dict with baseline and Fisher metrics.
    """
    precomputed = _precompute_model_data(
        model_path,
        dataset_loaders,
        layer_idx,
        t_target,
        seed,
    )

    return {
        "baseline": precomputed["baseline_metrics"],
        "fisher": _evaluate_fisher(precomputed),
    }


def run(
    dataset,
    *,
    force=False,
    seeds=None,
    t_obj_filter=None,
):
    from tqdm import tqdm

    train_loader, val_loader = create_dataset(dataset)

    base_dir = f"logs/{dataset}/threshold_research"
    models = _find_models(base_dir)
    if seeds:
        models = [(p, t, s) for p, t, s in models if s in seeds]
    if t_obj_filter:
        models = [(p, t, s) for p, t, s in models if t in t_obj_filter]
    if not models:
        logger.warning("No models found under %s", base_dir)
        return

    all_results = []

    with tqdm(total=len(models), desc="Fisher evaluation") as pbar:
        for model_path, t_obj, seed in models:
            pbar.set_postfix_str(f"t_obj={t_obj} seed={seed}")
            seed_dir = os.path.dirname(model_path)
            output_path = os.path.join(seed_dir, "fisher_results.json")

            if not force and os.path.exists(output_path):
                tqdm.write(f"  skip t_obj={t_obj} seed={seed} (already complete)")
                with open(output_path) as f:
                    result = json.load(f)
                all_results.append((seed, result))
                pbar.update(1)
                continue

            # Precompute once per model (the expensive part)
            def _set_step(step):
                pbar.set_postfix_str(f"t_obj={t_obj} seed={seed} | {step}")

            precomputed = _precompute_model_data(
                model_path,
                (train_loader, val_loader),
                0,
                t_obj,
                seed,
                status_fn=_set_step,
            )

            _set_step("evaluating fisher")
            result = {
                "baseline": precomputed["baseline_metrics"],
                "fisher": _evaluate_fisher(precomputed),
            }

            # Merge perturbation-optimal if available
            optimal_path = os.path.join(seed_dir, "optimal_thresholds.json")
            if os.path.exists(optimal_path):
                with open(optimal_path) as f:
                    optimal_data = json.load(f)
                result["perturbation_optimal"] = optimal_data.get(
                    "optimal_combined", optimal_data.get("baseline", {})
                )

            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)

            all_results.append((seed, result))

            # Print per-seed summary
            baseline_acc = result["baseline"]["accuracy"]
            fisher_acc = result["fisher"]["accuracy"]
            opt_acc = result.get("perturbation_optimal", {}).get("accuracy", "N/A")
            msg = f"    baseline={baseline_acc:.4f}  fisher={fisher_acc:.4f}"
            if isinstance(opt_acc, float):
                msg += f"  optimal={opt_acc:.4f}"
            else:
                msg += f"  optimal={opt_acc}"
            tqdm.write(msg)
            pbar.update(1)

    # Summary table
    if all_results:
        _print_summary_table(all_results)


def _print_summary_table(all_results: list[tuple[int, dict]]) -> None:
    """Print a formatted summary table across all seeds."""
    col_width = 10
    columns = ["baseline", "fisher", "optimal"]
    header = f"{'seed':>6}"
    for col in columns:
        header += f"  {col:>{col_width}}"
    logger.info("\n%s\n%s", header, "-" * len(header))

    accs: dict[str, list[float]] = {col: [] for col in columns}

    for seed, result in all_results:
        baseline_acc = result["baseline"]["accuracy"]
        accs["baseline"].append(baseline_acc)
        row = f"{seed:>6}  {baseline_acc:>{col_width}.4f}"

        fisher_acc = result.get("fisher", {}).get("accuracy")
        if fisher_acc is not None:
            accs["fisher"].append(fisher_acc)
            row += f"  {fisher_acc:>{col_width}.4f}"
        else:
            row += f"  {'N/A':>{col_width}}"

        opt_acc = result.get("perturbation_optimal", {}).get("accuracy")
        if opt_acc is not None:
            accs["optimal"].append(opt_acc)
            row += f"  {opt_acc:>{col_width}.4f}"
        else:
            row += f"  {'N/A':>{col_width}}"

        logger.info(row)

    # Mean row
    mean_row = f"{'-' * len(header)}\n{'mean':>6}"
    for col in columns:
        vals = accs[col]
        if vals:
            mean_row += f"  {np.mean(vals):>{col_width}.4f}"
        else:
            mean_row += f"  {'N/A':>{col_width}}"
    logger.info(mean_row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate Fisher discriminant threshold optimization"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="specific seeds to run"
    )
    parser.add_argument(
        "--t-obj",
        type=float,
        nargs="+",
        default=None,
        help="specific t_obj values to run (default: all)",
    )
    args = parser.parse_args()
    run(
        args.dataset,
        force=args.force,
        seeds=args.seeds,
        t_obj_filter=args.t_obj,
    )
