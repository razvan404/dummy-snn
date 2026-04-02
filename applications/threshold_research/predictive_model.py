import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def compute_predictive_model(
    optimal_deltas: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> dict:
    """Fit linear regression from metrics to optimal_deltas.

    :param optimal_deltas: Array of optimal threshold shifts per neuron.
    :param metrics: Dict mapping metric names to per-neuron arrays.
    :returns: Dict with r_squared and per-metric coefficients.
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

    :param optimal_deltas: Array of optimal threshold shifts per neuron.
    :param metrics: Dict mapping metric names to per-neuron arrays.
    :param n_folds: Number of cross-validation folds.
    :returns: Dict with linear_cv_r2, gbr_cv_r2, their stds, n_samples, n_features.
    """
    optimal_deltas = np.asarray(optimal_deltas)
    names = list(metrics.keys())
    X = np.column_stack([np.asarray(metrics[k]) for k in names])

    linear = LinearRegression()
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)

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


def denoise_optimal_deltas(
    accuracy_matrix: np.ndarray,
    perturbation_fractions: np.ndarray,
    original_thresholds: np.ndarray,
    method: str = "gaussian",
    sigma: float = 2.0,
) -> dict:
    """Denoise per-neuron accuracy curves before taking argmax.

    :param accuracy_matrix: (num_neurons, num_fractions)
    :param perturbation_fractions: (num_fractions,) fraction values.
    :param original_thresholds: (num_neurons,) original threshold per neuron.
    :param method: "gaussian" or "polynomial".
    :param sigma: Smoothing width for gaussian method.
    :returns: Dict with optimal_deltas (num_neurons,) denoised optimal threshold shifts,
        and confidence (num_neurons,) per-neuron confidence (peak - mean, normalized).
    """
    accuracy_matrix = np.asarray(accuracy_matrix)
    perturbation_fractions = np.asarray(perturbation_fractions)
    original_thresholds = np.asarray(original_thresholds)
    num_neurons = accuracy_matrix.shape[0]

    optimal_deltas = np.zeros(num_neurons)
    confidence = np.zeros(num_neurons)

    for n in range(num_neurons):
        curve = accuracy_matrix[n]

        if method == "gaussian":
            smoothed = gaussian_filter1d(curve, sigma=sigma)
            best_idx = np.argmax(smoothed)
            best_frac = perturbation_fractions[best_idx]
        elif method == "polynomial":
            coeffs = np.polyfit(perturbation_fractions, curve, deg=2)
            # vertex of ax^2 + bx + c is at x = -b/(2a)
            a, b, _c = coeffs
            if a < 0:  # concave down → real peak
                vertex_frac = -b / (2 * a)
                # Clamp to fraction range
                vertex_frac = np.clip(
                    vertex_frac,
                    perturbation_fractions[0],
                    perturbation_fractions[-1],
                )
                best_frac = vertex_frac
            else:
                # Concave up → take endpoint with higher value
                best_idx = np.argmax(curve)
                best_frac = perturbation_fractions[best_idx]
        else:
            raise ValueError(f"Unknown method: {method}")

        optimal_deltas[n] = best_frac * original_thresholds[n]

        # Confidence: peak accuracy minus mean accuracy
        peak_acc = np.max(curve)
        mean_acc = np.mean(curve)
        confidence[n] = peak_acc - mean_acc

    return {"optimal_deltas": optimal_deltas, "confidence": confidence}


def filter_sensitive_neurons(
    accuracy_matrix: np.ndarray,
    min_range: float = 0.005,
) -> np.ndarray:
    """Return boolean mask of neurons whose accuracy range exceeds min_range.

    :param accuracy_matrix: (num_neurons, num_fractions)
    :param min_range: Minimum accuracy range to be considered sensitive.
    :returns: (num_neurons,) boolean mask.
    """
    accuracy_matrix = np.asarray(accuracy_matrix)
    ranges = np.max(accuracy_matrix, axis=1) - np.min(accuracy_matrix, axis=1)
    return ranges > min_range
