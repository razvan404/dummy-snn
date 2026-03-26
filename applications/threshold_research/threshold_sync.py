import numpy as np
import torch
from sklearn.svm import LinearSVC

from applications.threshold_research.neuron_perturbation import (
    compute_features_with_thresholds,
    spike_times_from_potentials,
)
from spiking.evaluation.feature_extraction import spike_times_to_features


def compute_fisher_thresholds(
    cum_potentials: torch.Tensor,
    boundary_times: torch.Tensor,
    original_thresholds: torch.Tensor,
    labels: np.ndarray,
    t_target: float,
    n_fractions: int = 31,
    frac_min: float = -0.5,
    frac_max: float = 0.25,
) -> torch.Tensor:
    """Find per-neuron thresholds that maximize train accuracy.

    Fits a baseline LinearSVC on original features, then for each neuron searches
    over threshold multipliers (1 + frac). For each candidate, swaps that neuron's
    feature column and measures train accuracy with the baseline classifier.
    Tie-break: keep original (frac=0).

    cum_potentials: (B, O, G) — precomputed cumulative membrane potentials.
    boundary_times: (G,) — sorted unique input spike times.
    original_thresholds: (O,) — current thresholds.
    labels: (B,) — integer class labels.
    t_target: target spike time for feature computation.
    n_fractions: number of multipliers to search.
    frac_min: minimum fraction (inclusive).
    frac_max: maximum fraction (inclusive).

    Returns: (O,) adjusted thresholds.
    """
    O = original_thresholds.shape[0]
    fractions = np.linspace(frac_min, frac_max, n_fractions)

    # Compute baseline features and fit classifier
    baseline_features = compute_features_with_thresholds(
        cum_potentials, boundary_times, original_thresholds, t_target
    )
    clf = LinearSVC(dual=False, tol=1e-3, max_iter=10000)
    clf.fit(baseline_features, labels)
    baseline_accuracy = (clf.predict(baseline_features) == labels).mean()

    new_thresholds = original_thresholds.clone()

    for j in range(O):
        best_accuracy = baseline_accuracy
        best_frac = 0.0

        for frac in fractions:
            threshold_j = original_thresholds[j].item() * (1.0 + frac)
            spike_times_j = spike_times_from_potentials(
                cum_potentials[:, j, :], boundary_times, threshold_j
            )
            features_j = spike_times_to_features(spike_times_j, t_target).numpy()

            # Swap column j in baseline features and measure accuracy
            modified_features = baseline_features.copy()
            modified_features[:, j] = features_j
            accuracy = (clf.predict(modified_features) == labels).mean()

            # Tie-break: prefer frac=0 (original) when equal
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and abs(frac) < abs(best_frac)
            ):
                best_accuracy = accuracy
                best_frac = frac

        if best_frac != 0.0:
            new_thresholds[j] = original_thresholds[j] * (1.0 + best_frac)

    return new_thresholds
