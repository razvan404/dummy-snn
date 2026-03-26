import argparse
import json
import logging

import numpy as np
from sklearn.linear_model import RidgeClassifier

logger = logging.getLogger(__name__)

from applications.threshold_research.analysis import correlations_report
from applications.threshold_research.metrics import compute_classifier_metrics
from applications.threshold_research.predictive_model import (
    compute_predictive_model,
    denoise_optimal_deltas,
    filter_sensitive_neurons,
)


def load_seed_data(seed_dir: str) -> dict:
    """Load all results and cached features for a single seed."""
    with open(f"{seed_dir}/perturbation_results.json") as f:
        perturbation = json.load(f)

    with open(f"{seed_dir}/training_metrics.json") as f:
        training = json.load(f)

    with open(f"{seed_dir}/post_hoc_metrics.json") as f:
        post_hoc = json.load(f)

    cache = f"{seed_dir}/perturbation_cache"
    X_train = np.load(f"{cache}/baseline_train.npy")
    X_val = np.load(f"{cache}/baseline_val.npy")
    y_train = np.load(f"{cache}/labels_train.npy")
    y_val = np.load(f"{cache}/labels_val.npy")

    return {
        "perturbation": perturbation,
        "training_metrics": training,
        "post_hoc_metrics": post_hoc,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def print_section(title: str) -> None:
    logger.info("\n%s\n  %s\n%s", "=" * 60, title, "=" * 60)


def print_confidence_distribution(confidence: np.ndarray, label: str) -> None:
    """Print percentile distribution of confidence values."""
    pcts = [0, 25, 50, 75, 90, 95, 100]
    vals = np.percentile(confidence, pcts)
    logger.info("\n  %s confidence distribution:", label)
    logger.info("  %12s  %10s", "Percentile", "Value")
    for p, v in zip(pcts, vals):
        logger.info("  %11d%%  %10.6f", p, v)


def print_correlations(report: dict) -> None:
    """Print correlation results as a table."""
    logger.info("\n  %-45s %8s %10s", "Metric", "r", "p")
    logger.info("  %s %s %s", "-" * 45, "-" * 8, "-" * 10)
    for key, val in report.items():
        if "r" in val and "p" in val:
            sig = "*" if val["p"] < 0.05 else " "
            logger.info("  %-45s %8.4f %10.2e %s", key, val["r"], val["p"], sig)
        elif "increase" in val:
            logger.info(
                "  %-45s +%s / -%s / =%s",
                key,
                val["increase"],
                val["decrease"],
                val["same"],
            )


def print_predictive_comparison(results: list[tuple[str, dict]]) -> None:
    """Print R-squared comparison table."""
    logger.info("\n  %-55s %8s", "Configuration", "R²")
    logger.info("  %s %s", "-" * 55, "-" * 8)
    for label, result in results:
        logger.info("  %-55s %8.4f", label, result["r_squared"])


def run(seed_dir: str) -> None:
    data = load_seed_data(seed_dir)
    perturbation = data["perturbation"]

    accuracy_matrix = np.array(perturbation["accuracy_matrix"])
    perturbation_fractions = np.array(perturbation["perturbation_fractions"])
    original_thresholds = np.array(perturbation["original_thresholds"])
    raw_deltas = np.array(perturbation["optimal_deltas"])
    num_neurons = len(raw_deltas)

    logger.info("Seed directory: %s", seed_dir)
    logger.info("Neurons: %d, Fractions: %d", num_neurons, len(perturbation_fractions))
    baseline = perturbation["baseline"]
    if isinstance(baseline, dict):
        logger.info("Baseline accuracy: %.4f", baseline["accuracy"])
    else:
        logger.info("Baseline accuracy: %.4f", baseline)

    # --- Step 1: Fit classifier ---
    print_section("Classifier Fit")
    clf = RidgeClassifier()
    clf.fit(data["X_train"], data["y_train"])
    train_acc = clf.score(data["X_train"], data["y_train"])
    val_acc = clf.score(data["X_val"], data["y_val"])
    logger.info("  RidgeClassifier train acc: %.4f, val acc: %.4f", train_acc, val_acc)

    # --- Step 2: Denoising ---
    print_section("Denoising Optimal Deltas")

    gaussian = denoise_optimal_deltas(
        accuracy_matrix,
        perturbation_fractions,
        original_thresholds,
        method="gaussian",
        sigma=2.0,
    )
    polynomial = denoise_optimal_deltas(
        accuracy_matrix,
        perturbation_fractions,
        original_thresholds,
        method="polynomial",
    )

    print_confidence_distribution(gaussian["confidence"], "Gaussian")
    print_confidence_distribution(polynomial["confidence"], "Polynomial")

    # Compare raw vs denoised deltas
    raw_vs_gauss = np.corrcoef(raw_deltas, gaussian["optimal_deltas"])[0, 1]
    raw_vs_poly = np.corrcoef(raw_deltas, polynomial["optimal_deltas"])[0, 1]
    logger.info("  Raw vs gaussian deltas correlation: %.4f", raw_vs_gauss)
    logger.info("  Raw vs polynomial deltas correlation: %.4f", raw_vs_poly)

    # --- Step 3: Filter sensitive neurons ---
    print_section("Sensitive Neuron Filtering")

    for threshold in [0.001, 0.005, 0.01, 0.02]:
        mask = filter_sensitive_neurons(accuracy_matrix, min_range=threshold)
        logger.info(
            "  min_range=%.3f: %3d/%d sensitive neurons",
            threshold,
            mask.sum(),
            num_neurons,
        )

    sensitive_mask = filter_sensitive_neurons(accuracy_matrix, min_range=0.001)

    # --- Step 4: Classifier metrics ---
    print_section("Classifier Metrics")

    classifier_metrics = compute_classifier_metrics(
        clf,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
    )

    for key, vals in classifier_metrics.items():
        vals = np.asarray(vals)
        finite = vals[np.isfinite(vals)]
        logger.info(
            "  %-40s mean=%10.4f  std=%10.4f  min=%10.4f  max=%10.4f",
            key,
            finite.mean(),
            finite.std(),
            finite.min(),
            finite.max(),
        )

    # --- Step 5: Correlations ---
    print_section("Correlations Report")

    report = correlations_report(
        perturbation,
        training_metrics=data["training_metrics"],
        post_hoc_metrics=data["post_hoc_metrics"],
        classifier_metrics=classifier_metrics,
    )
    print_correlations(report)

    # --- Step 6: Predictive models ---
    print_section("Predictive Model Comparison")

    post_hoc = {k: np.array(v) for k, v in data["post_hoc_metrics"].items()}
    # Handle inf in avg_spike_time
    if "avg_spike_time" in post_hoc:
        ast = post_hoc["avg_spike_time"]
        ast[~np.isfinite(ast)] = np.nanmax(ast[np.isfinite(ast)])

    # Handle inf in fisher_discriminant_ratio
    classifier_arrays = {k: np.array(v) for k, v in classifier_metrics.items()}
    if "fisher_discriminant_ratio" in classifier_arrays:
        fdr = classifier_arrays["fisher_discriminant_ratio"]
        fdr[~np.isfinite(fdr)] = np.nanmax(fdr[np.isfinite(fdr)])

    combined = {**post_hoc, **classifier_arrays}

    configs = []

    # All neurons, post-hoc only, raw deltas
    configs.append(
        (
            "Post-hoc only → raw deltas (all neurons)",
            compute_predictive_model(raw_deltas, post_hoc),
        )
    )

    # All neurons, post-hoc + classifier, raw deltas
    configs.append(
        (
            "Post-hoc + classifier → raw deltas (all neurons)",
            compute_predictive_model(raw_deltas, combined),
        )
    )

    # All neurons, post-hoc + classifier, gaussian denoised deltas
    configs.append(
        (
            "Post-hoc + classifier → gaussian deltas (all neurons)",
            compute_predictive_model(gaussian["optimal_deltas"], combined),
        )
    )

    # Sensitive neurons only
    if sensitive_mask.sum() > 10:
        s = sensitive_mask
        post_hoc_s = {k: v[s] for k, v in post_hoc.items()}
        combined_s = {k: v[s] for k, v in combined.items()}

        configs.append(
            (
                f"Post-hoc only → raw deltas ({s.sum()} sensitive)",
                compute_predictive_model(raw_deltas[s], post_hoc_s),
            )
        )

        configs.append(
            (
                f"Post-hoc + classifier → raw deltas ({s.sum()} sensitive)",
                compute_predictive_model(raw_deltas[s], combined_s),
            )
        )

        configs.append(
            (
                f"Post-hoc + classifier → gaussian deltas ({s.sum()} sensitive)",
                compute_predictive_model(gaussian["optimal_deltas"][s], combined_s),
            )
        )

    print_predictive_comparison(configs)

    # --- Top coefficients in best model ---
    print_section("Top Coefficients (Post-hoc + classifier → gaussian, all neurons)")

    best = compute_predictive_model(gaussian["optimal_deltas"], combined)
    coeffs = best["coefficients"]
    sorted_coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info("\n  %-40s %12s", "Feature", "Coefficient")
    logger.info("  %s %s", "-" * 40, "-" * 12)
    for name, coef in sorted_coeffs:
        logger.info("  %-40s %12.6f", name, coef)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Validate classifier-aware metrics on real perturbation data"
    )
    parser.add_argument(
        "--seed-dir",
        default="logs/mnist/threshold_research/tobj_0.875/seed_1",
        help="Path to a single seed's results directory",
    )
    args = parser.parse_args()
    run(args.seed_dir)
