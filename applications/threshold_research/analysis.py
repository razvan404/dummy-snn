import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

from spiking import load_model
from spiking.layers import SpikingSequential


def compute_threshold_sensitivity(accuracy_matrix: np.ndarray) -> np.ndarray:
    """Per-neuron sensitivity: std of accuracy across perturbation fractions.

    accuracy_matrix: (num_neurons, num_fractions)
    Returns: (num_neurons,)
    """
    return np.std(accuracy_matrix, axis=1)


def compute_feature_importance(classifier, X: np.ndarray) -> np.ndarray:
    """Extract per-feature importance from a fitted LinearSVC.

    For multi-class, returns mean absolute coefficient across classes.
    Returns: (num_features,)
    """
    coefs = np.abs(classifier.coef_)
    return np.mean(coefs, axis=0)


def compute_post_hoc_metrics(
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
) -> dict:
    """Compute per-neuron metrics from saved model without retraining.

    Returns dict with per-neuron arrays:
      weight_l2_norm, weight_l1_norm, avg_spike_time, spike_rate, weight_std.
    """
    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    sub_model.eval()

    weights = layer.weights.detach()
    num_outputs = layer.num_outputs

    weight_l2_norm = torch.norm(weights, p=2, dim=1).numpy().tolist()
    weight_l1_norm = torch.norm(weights, p=1, dim=1).numpy().tolist()
    weight_std = weights.std(dim=1).numpy().tolist()

    # Collect spike times and membrane potentials across training set
    train_loader = dataset_loaders[0]
    batched_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=False)

    thresholds = layer.thresholds.detach()
    preceding_layers = model.layers[:layer_idx]

    spike_time_sum = np.zeros(num_outputs)
    spike_count = np.zeros(num_outputs)
    nonspiking_count = np.zeros(num_outputs)
    ratio_sum_nonspiking = np.zeros(num_outputs)
    ratio_max_nonspiking = np.zeros(num_outputs)
    ratio_all = []  # collect per-sample ratios for std computation
    total_samples = 0

    with torch.no_grad():
        for batch_times, _labels in batched_loader:
            flat_input = batch_times.flatten(1)

            # Propagate through preceding layers if multi-layer
            layer_input = flat_input
            for prev_layer in preceding_layers:
                layer_input = prev_layer.infer_spike_times_batch(layer_input)

            st, cum_pot = layer.infer_spike_times_and_potentials_batch(layer_input)
            finite_mask = torch.isfinite(st)

            st_np = st.numpy()
            finite_np = finite_mask.numpy()
            ratio = (cum_pot / thresholds).numpy()  # (B, num_outputs)

            batch_size = st.shape[0]
            total_samples += batch_size
            ratio_all.append(ratio)

            for n in range(num_outputs):
                finite_n = finite_np[:, n]
                spike_count[n] += finite_n.sum()
                spike_time_sum[n] += st_np[finite_n, n].sum()

                nonspiking_n = ~finite_n
                n_nonspiking = nonspiking_n.sum()
                nonspiking_count[n] += n_nonspiking
                if n_nonspiking > 0:
                    ratios_n = ratio[nonspiking_n, n]
                    ratio_sum_nonspiking[n] += ratios_n.sum()
                    ratio_max_nonspiking[n] = max(
                        ratio_max_nonspiking[n], ratios_n.max()
                    )

    avg_spike_time = np.where(
        spike_count > 0, spike_time_sum / spike_count, float("inf")
    ).tolist()
    spike_rate = (spike_count / total_samples).tolist()

    with np.errstate(invalid="ignore"):
        potential_ratio_mean = np.where(
            nonspiking_count > 0, ratio_sum_nonspiking / nonspiking_count, 0.0
        ).tolist()
    potential_ratio_max = ratio_max_nonspiking.tolist()

    all_ratios = np.concatenate(ratio_all, axis=0)  # (total_samples, num_outputs)
    potential_ratio_std = np.std(all_ratios, axis=0).tolist()

    return {
        "weight_l2_norm": weight_l2_norm,
        "weight_l1_norm": weight_l1_norm,
        "avg_spike_time": avg_spike_time,
        "spike_rate": spike_rate,
        "weight_std": weight_std,
        "potential_ratio_mean": potential_ratio_mean,
        "potential_ratio_max": potential_ratio_max,
        "potential_ratio_std": potential_ratio_std,
    }


def compute_predictive_model(
    optimal_deltas: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict:
    """Fit linear regression from metrics to optimal_deltas.

    Returns dict with r_squared and per-metric coefficients.
    """
    optimal_deltas = np.asarray(optimal_deltas)
    names = list(metrics.keys())
    X = np.column_stack([np.asarray(metrics[k]) for k in names])

    reg = LinearRegression().fit(X, optimal_deltas)
    r_squared = reg.score(X, optimal_deltas)

    return {
        "r_squared": float(r_squared),
        "coefficients": {name: float(c) for name, c in zip(names, reg.coef_)},
        "intercept": float(reg.intercept_),
    }


def compute_nonlinear_predictive_model(
    optimal_deltas: np.ndarray,
    metrics: dict[str, np.ndarray],
    n_folds: int = 5,
) -> dict:
    """Compare linear vs gradient boosting regression for predicting optimal deltas.

    Cross-validated R² for fair comparison (GBR can overfit on small sample sizes).
    Returns dict with linear_cv_r2, gbr_cv_r2, their stds, n_samples, n_features.
    """
    optimal_deltas = np.asarray(optimal_deltas)
    names = list(metrics.keys())
    X = np.column_stack([np.asarray(metrics[k]) for k in names])

    linear = LinearRegression()
    gbr = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, random_state=0
    )

    linear_scores = cross_val_score(linear, X, optimal_deltas, cv=n_folds, scoring="r2")
    gbr_scores = cross_val_score(gbr, X, optimal_deltas, cv=n_folds, scoring="r2")

    return {
        "linear_cv_r2": float(np.mean(linear_scores)),
        "linear_cv_r2_std": float(np.std(linear_scores)),
        "gbr_cv_r2": float(np.mean(gbr_scores)),
        "gbr_cv_r2_std": float(np.std(gbr_scores)),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


def correlations_report(
    results: dict,
    winner_counts: np.ndarray | None = None,
    training_metrics: dict | None = None,
    post_hoc_metrics: dict | None = None,
) -> dict:
    """Compute correlations between threshold properties and performance metrics.

    results: output of run_perturbation_sweep().
    winner_counts: optional (num_neurons,) array of competition win counts.
    training_metrics: optional dict with spike_counts, update_counts, threshold_drift.
    post_hoc_metrics: optional dict with weight_l2_norm, weight_l1_norm, etc.

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
