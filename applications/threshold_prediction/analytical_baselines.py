import numpy as np
from tqdm import tqdm

from .cached_coordinate_descent import CachedThresholdOptimizer


def quantile_thresholds(V_max: np.ndarray, p: float) -> np.ndarray:
    """Set threshold to (1-p)-quantile of V_max per neuron.

    Args:
        V_max: (N, num_neurons) peak membrane potentials.
        p: Target firing probability. Higher p → lower threshold.

    Returns:
        (num_neurons,) threshold array.
    """
    return np.quantile(V_max, 1.0 - p, axis=0)


def mean_plus_k_sigma(V_max: np.ndarray, k: float) -> np.ndarray:
    """Set threshold to mean + k * std of V_max per neuron.

    Args:
        V_max: (N, num_neurons) peak membrane potentials.
        k: Number of standard deviations above mean.

    Returns:
        (num_neurons,) threshold array.
    """
    return V_max.mean(axis=0) + k * V_max.std(axis=0)


def otsu_per_neuron(V_max: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Apply Otsu's method to each neuron's V_max distribution.

    Finds the threshold maximizing inter-class variance (bimodal split).

    Args:
        V_max: (N, num_neurons) peak membrane potentials.
        n_bins: Number of histogram bins.

    Returns:
        (num_neurons,) threshold array.
    """
    num_neurons = V_max.shape[1]
    thresholds = np.zeros(num_neurons)

    for i in range(num_neurons):
        col = V_max[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-10:
            thresholds[i] = col_min
            continue

        counts, bin_edges = np.histogram(col, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        total = counts.sum()

        if total == 0:
            thresholds[i] = col_min
            continue

        best_variance = -1.0
        best_threshold = col_min

        cum_sum_0 = 0.0
        cum_count_0 = 0.0

        total_sum = (counts * bin_centers).sum()

        for j in range(n_bins - 1):
            cum_count_0 += counts[j]
            cum_sum_0 += counts[j] * bin_centers[j]

            w0 = cum_count_0 / total
            w1 = 1.0 - w0

            if w0 < 1e-10 or w1 < 1e-10:
                continue

            mean0 = cum_sum_0 / cum_count_0
            mean1 = (total_sum - cum_sum_0) / (total - cum_count_0)

            variance = w0 * w1 * (mean0 - mean1) ** 2

            if variance > best_variance:
                best_variance = variance
                best_threshold = bin_edges[j + 1]

        thresholds[i] = best_threshold

    return thresholds


def uniform_mean(trained_thresholds: np.ndarray) -> np.ndarray:
    """All neurons get the mean of the trained thresholds."""
    return np.full_like(trained_thresholds, trained_thresholds.mean())


def evaluate_all_baselines(
    optimizer: CachedThresholdOptimizer,
    V_max: np.ndarray,
    trained_thresholds: np.ndarray,
) -> list[dict]:
    """Run all analytical baselines and return comparison results.

    Returns list of dicts with keys: method, params, accuracy.
    """
    results = []

    configs = []

    # Quantile baselines
    for p in [0.001, 0.002, 0.004, 0.005, 0.01, 0.02, 0.05]:
        configs.append(
            ("quantile", {"p": p}, lambda p=p: quantile_thresholds(V_max, p))
        )

    # Mean + k*sigma baselines
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        configs.append(
            ("mean_plus_k_sigma", {"k": k}, lambda k=k: mean_plus_k_sigma(V_max, k))
        )

    # Otsu
    configs.append(("otsu", {}, lambda: otsu_per_neuron(V_max)))

    # Uniform mean
    configs.append(("uniform_mean", {}, lambda: uniform_mean(trained_thresholds)))

    # Trained thresholds (baseline reference)
    configs.append(("trained", {}, lambda: trained_thresholds.copy()))

    pbar = tqdm(configs, desc="Evaluating baselines", leave=False)
    for method, params, threshold_fn in pbar:
        pbar.set_postfix_str(f"{method} {params}")
        thresholds = threshold_fn()
        # Ensure minimum threshold
        thresholds = np.maximum(thresholds, 0.1)
        acc = optimizer.evaluate_thresholds_with_refit(thresholds)
        results.append(
            {
                "method": method,
                "params": params,
                "accuracy": acc,
                "thresholds_mean": float(thresholds.mean()),
                "thresholds_std": float(thresholds.std()),
            }
        )

    return results
