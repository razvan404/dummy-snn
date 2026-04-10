import numpy as np

ORDERINGS = [
    "descending_importance",
    "ascending_importance",
    "early_spike",
    "late_spike",
    "hybrid_importance",
    "hybrid_spike_time",
    # Threshold-deviation orderings (|threshold_i - mean(thresholds)|, no log needed)
    "high_abs_drift",  # largest deviation from mean threshold first
    "low_abs_drift",  # smallest deviation from mean threshold first
    "hybrid_abs_drift",  # interleaved: most deviated ↔ least deviated
    # Training-log-based orderings (require mean_spike_time_per_neuron in training_metrics.json)
    "training_early_spike",  # earliest mean spike time during training first
    "training_late_spike",  # latest mean spike time during training first
    "hybrid_training_spike",  # interleaved: earliest ↔ latest training spike time
]


def _interleave(indices: np.ndarray) -> np.ndarray:
    """Interleave from both ends: [0, -1, 1, -2, 2, -3, ...]."""
    result = []
    lo, hi = 0, len(indices) - 1
    toggle = True
    while lo <= hi:
        result.append(indices[lo] if toggle else indices[hi])
        if toggle:
            lo += 1
        else:
            hi -= 1
        toggle = not toggle
    return np.array(result)


def get_filter_order(
    ordering: str,
    filter_importance: np.ndarray,
    mean_spike_times: np.ndarray,
    *,
    threshold_drift: np.ndarray | None = None,
    training_spike_times: np.ndarray | None = None,
) -> np.ndarray:
    """Return filter indices in the specified processing order.

    :param ordering: One of ORDERINGS.
    :param filter_importance: Per-filter Ridge coefficient magnitude sum.
    :param mean_spike_times: Per-filter mean spike time over the training set.
    :param threshold_drift: Per-filter deviation from the mean threshold,
        i.e. |threshold_i - mean(thresholds)|. Required for high_abs_drift /
        low_abs_drift. Computed directly from the model's current thresholds.
    :param training_spike_times: Per-filter mean spike time recorded *during*
        STDP training (from training_metrics.json). Required for
        training_early_spike / training_late_spike. May contain NaN for filters
        that never fired; those are placed last.
    :returns: 1-D array of filter indices in the order they should be processed.
    """
    asc_imp = np.argsort(filter_importance)
    asc_spike = np.argsort(mean_spike_times)
    match ordering:
        case "descending_importance":
            return asc_imp[::-1]
        case "ascending_importance":
            return asc_imp
        case "early_spike":
            return asc_spike
        case "late_spike":
            return asc_spike[::-1]
        case "hybrid_importance":
            return _interleave(asc_imp[::-1])
        case "hybrid_spike_time":
            return _interleave(asc_spike)
        case "high_abs_drift" | "low_abs_drift" | "hybrid_abs_drift":
            if threshold_drift is None:
                raise ValueError(
                    f"ordering={ordering!r} requires threshold_drift "
                    "(load training_metrics.json alongside the model)"
                )
            abs_drift = np.abs(threshold_drift)
            asc_drift = np.argsort(abs_drift)
            if ordering == "low_abs_drift":
                return asc_drift
            elif ordering == "high_abs_drift":
                return asc_drift[::-1]
            else:
                return _interleave(asc_drift[::-1])
        case "training_early_spike" | "training_late_spike" | "hybrid_training_spike":
            if training_spike_times is None:
                raise ValueError(
                    f"ordering={ordering!r} requires training_spike_times "
                    "(needs mean_spike_time_per_neuron in training_metrics.json — "
                    "retrain with an updated train_models.py)"
                )
            # NaN filters (never fired during training) go last
            nan_mask = np.isnan(training_spike_times)
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            asc_train = valid_idx[np.argsort(training_spike_times[valid_idx])]
            if ordering == "training_early_spike":
                return np.concatenate([asc_train, nan_idx])
            elif ordering == "training_late_spike":
                return np.concatenate([asc_train[::-1], nan_idx])
            else:
                return np.concatenate([_interleave(asc_train), nan_idx])
        case _:
            raise ValueError(f"Unknown ordering: {ordering!r}. Choose from {ORDERINGS}")
