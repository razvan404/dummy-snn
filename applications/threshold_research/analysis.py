import numpy as np
from scipy import stats

from .metrics import compute_threshold_sensitivity


def correlations_report(
    results: dict,
    winner_counts: np.ndarray | None = None,
    training_metrics: dict | None = None,
    post_hoc_metrics: dict | None = None,
    classifier_metrics: dict | None = None,
) -> dict:
    """Compute correlations between threshold properties and performance metrics.

    results: output of run_perturbation_sweep().
    winner_counts: optional (num_neurons,) array of competition win counts.
    training_metrics: optional dict with spike_counts, update_counts, threshold_drift.
    post_hoc_metrics: optional dict with weight_l2_norm, weight_l1_norm, etc.
    classifier_metrics: optional dict from compute_classifier_metrics().

    Returns dict of {correlation_name: {"r": ..., "p": ..., "description": ...}}.
    """
    original = np.array(results["original_thresholds"])
    optimal_deltas = np.array(results["optimal_deltas"])
    accuracy_matrix = np.array(results["accuracy_matrix"])
    sensitivity = compute_threshold_sensitivity(accuracy_matrix)

    report = {}

    # Original threshold vs optimal delta
    r, p = stats.pearsonr(original, optimal_deltas)
    report["threshold_vs_delta"] = {
        "r": float(r),
        "p": float(p),
        "description": "Original threshold vs optimal threshold shift",
    }

    # Absolute threshold vs sensitivity
    r, p = stats.pearsonr(original, sensitivity)
    report["threshold_vs_sensitivity"] = {
        "r": float(r),
        "p": float(p),
        "description": "Original threshold vs perturbation sensitivity",
    }

    # Optimal delta direction (fraction of neurons that need increase vs decrease)
    n_increase = np.sum(optimal_deltas > 0)
    n_decrease = np.sum(optimal_deltas < 0)
    n_same = np.sum(optimal_deltas == 0)
    report["delta_direction"] = {
        "increase": int(n_increase),
        "decrease": int(n_decrease),
        "same": int(n_same),
        "description": "Count of neurons needing threshold increase/decrease/no change",
    }

    if winner_counts is not None:
        winner_counts = np.asarray(winner_counts, dtype=float)

        r, p = stats.pearsonr(winner_counts, optimal_deltas)
        report["winners_vs_delta"] = {
            "r": float(r),
            "p": float(p),
            "description": "Competition win count vs optimal threshold shift",
        }

        r, p = stats.pearsonr(winner_counts, sensitivity)
        report["winners_vs_sensitivity"] = {
            "r": float(r),
            "p": float(p),
            "description": "Competition win count vs perturbation sensitivity",
        }

    def _add_correlation(key, values, description):
        values = np.asarray(values, dtype=float)
        r, p = stats.pearsonr(values, optimal_deltas)
        report[key] = {"r": float(r), "p": float(p), "description": description}

    if training_metrics is not None:
        if "spike_counts" in training_metrics:
            _add_correlation(
                "spike_count_vs_delta",
                training_metrics["spike_counts"],
                "Training spike count vs optimal threshold shift",
            )
        if "update_counts" in training_metrics:
            _add_correlation(
                "update_count_vs_delta",
                training_metrics["update_counts"],
                "STDP update count vs optimal threshold shift",
            )
        if "threshold_drift" in training_metrics:
            _add_correlation(
                "threshold_drift_vs_delta",
                training_metrics["threshold_drift"],
                "Threshold drift during training vs optimal threshold shift",
            )

    if post_hoc_metrics is not None:
        metric_pairs = [
            ("weight_l2_norm", "weight_l2_vs_delta", "Weight L2 norm"),
            ("weight_l1_norm", "weight_l1_vs_delta", "Weight L1 norm"),
            ("avg_spike_time", "avg_spike_time_vs_delta", "Average spike time"),
            ("spike_rate", "spike_rate_vs_delta", "Spike rate"),
            (
                "weight_std",
                "weight_std_vs_delta",
                "Weight std (receptive field sharpness)",
            ),
            (
                "potential_ratio_mean",
                "potential_ratio_mean_vs_delta",
                "Mean potential/threshold ratio (non-spiking)",
            ),
            (
                "potential_ratio_max",
                "potential_ratio_max_vs_delta",
                "Max potential/threshold ratio (non-spiking)",
            ),
            (
                "potential_ratio_std",
                "potential_ratio_std_vs_delta",
                "Potential/threshold ratio std",
            ),
        ]
        for metric_key, report_key, label in metric_pairs:
            if metric_key in post_hoc_metrics:
                _add_correlation(
                    report_key,
                    post_hoc_metrics[metric_key],
                    f"{label} vs optimal threshold shift",
                )

    if classifier_metrics is not None:
        classifier_pairs = [
            (
                "coef_magnitude",
                "coef_magnitude_vs_delta",
                "Classifier coefficient magnitude",
            ),
            (
                "misclassified_margin_contribution",
                "misclassified_margin_vs_delta",
                "Misclassified margin contribution",
            ),
            (
                "fisher_discriminant_ratio",
                "fisher_ratio_vs_delta",
                "Fisher discriminant ratio",
            ),
            (
                "max_feature_correlation",
                "max_correlation_vs_delta",
                "Max feature correlation (redundancy)",
            ),
        ]
        for metric_key, report_key, label in classifier_pairs:
            if metric_key in classifier_metrics:
                _add_correlation(
                    report_key,
                    classifier_metrics[metric_key],
                    f"{label} vs optimal threshold shift",
                )

    if "baseline_importance" in results and "optimal_importance" in results:
        baseline_imp = np.asarray(results["baseline_importance"], dtype=float)
        optimal_imp = np.asarray(results["optimal_importance"], dtype=float)
        importance_gap = optimal_imp - baseline_imp
        _add_correlation(
            "importance_gap_vs_delta",
            importance_gap,
            "Classifier importance change (optimal - baseline) vs optimal threshold shift",
        )

    return report
