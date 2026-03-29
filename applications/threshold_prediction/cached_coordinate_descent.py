import numpy as np
import torch
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from applications.threshold_research.neuron_perturbation import (
    collect_input_times,
    compute_features_with_thresholds,
    precompute_cumulative_potentials,
    spike_times_from_potentials,
)
from spiking.evaluation.eval_utils import compute_metrics
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.layers.integrate_and_fire import IntegrateAndFireLayer


class CachedThresholdOptimizer:
    """Efficient per-neuron threshold optimization using precomputed cumulative potentials.

    Key insight: membrane potentials don't depend on thresholds.
    Only spike times do. Precompute cumulative potentials once, then
    cheaply resolve spike times for any threshold configuration.

    Optimizes each neuron independently: for each neuron, find the
    threshold that maximizes accuracy when only that neuron's threshold
    is changed. This is fast (O(neurons × candidates) evaluations)
    and shows whether per-neuron optimization alone is sufficient.
    """

    def __init__(
        self,
        cum_train: torch.Tensor,
        cum_val: torch.Tensor,
        boundary_train: torch.Tensor,
        boundary_val: torch.Tensor,
        y_train: np.ndarray,
        y_val: np.ndarray,
        t_target: float | None,
        baseline_clf: LinearSVC,
        baseline_X_train: np.ndarray,
        baseline_X_val: np.ndarray,
    ):
        self.cum_train = cum_train
        self.cum_val = cum_val
        self.boundary_train = boundary_train
        self.boundary_val = boundary_val
        self.y_train = y_train
        self.y_val = y_val
        self.t_target = t_target
        self.baseline_clf = baseline_clf
        self.baseline_X_train = baseline_X_train
        self.baseline_X_val = baseline_X_val

    @classmethod
    def from_layer_and_data(
        cls,
        layer: IntegrateAndFireLayer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        t_target: float | None = None,
    ) -> "CachedThresholdOptimizer":
        """Build optimizer from a trained layer and data loaders."""
        weights = layer.weights.detach()

        tqdm.write("  Collecting input times...")
        train_input = collect_input_times(train_loader)
        val_input = collect_input_times(val_loader)

        tqdm.write("  Precomputing cumulative potentials...")
        cum_train, boundary_train = precompute_cumulative_potentials(
            train_input, weights
        )
        cum_val, boundary_val = precompute_cumulative_potentials(val_input, weights)

        # Collect labels
        y_train = np.array([label.item() for _, label in train_loader.dataset])
        y_val = np.array([label.item() for _, label in val_loader.dataset])

        # Compute baseline features with correct per-split boundaries
        thresholds = layer.thresholds.detach()
        X_train = compute_features_with_thresholds(
            cum_train, boundary_train, thresholds, t_target
        )
        X_val = compute_features_with_thresholds(
            cum_val, boundary_val, thresholds, t_target
        )

        baseline_clf = LinearSVC(dual=False, tol=1e-3, max_iter=10000)
        baseline_clf.fit(X_train, y_train)

        return cls(
            cum_train=cum_train,
            cum_val=cum_val,
            boundary_train=boundary_train,
            boundary_val=boundary_val,
            y_train=y_train,
            y_val=y_val,
            t_target=t_target,
            baseline_clf=baseline_clf,
            baseline_X_train=X_train,
            baseline_X_val=X_val,
        )

    def evaluate_thresholds_with_refit(
        self, thresholds: torch.Tensor | np.ndarray
    ) -> float:
        """Evaluate accuracy with a freshly-fitted classifier for new thresholds.

        Use this when evaluating thresholds that differ substantially from the
        originals (e.g. combined per-neuron optimals). Retrains the SVC on features
        computed from the new thresholds so the decision boundary matches the
        shifted feature distribution.
        """
        if isinstance(thresholds, np.ndarray):
            thresholds = torch.from_numpy(thresholds).float()

        X_train = compute_features_with_thresholds(
            self.cum_train, self.boundary_train, thresholds, self.t_target
        )
        X_val = compute_features_with_thresholds(
            self.cum_val, self.boundary_val, thresholds, self.t_target
        )
        clf = LinearSVC(dual=False, tol=1e-3, max_iter=10000)
        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_val)
        return compute_metrics(self.y_val, y_pred)["accuracy"]

    def evaluate_thresholds(self, thresholds: torch.Tensor | np.ndarray) -> float:
        """Evaluate validation accuracy for a given threshold vector."""
        if isinstance(thresholds, np.ndarray):
            thresholds = torch.from_numpy(thresholds).float()

        X_val = compute_features_with_thresholds(
            self.cum_val, self.boundary_val, thresholds, self.t_target
        )
        y_pred = self.baseline_clf.predict(X_val)
        return compute_metrics(self.y_val, y_pred)["accuracy"]

    def _eval_single_neuron_swap(self, neuron_idx: int, new_threshold: float) -> float:
        """Evaluate accuracy when only one neuron's threshold changes.

        Replaces that neuron's feature column in the cached val features
        and predicts with the frozen classifier. Very fast.
        """
        # Recompute only this neuron's spike times on val
        new_spike_times = spike_times_from_potentials(
            self.cum_val[:, neuron_idx, :],
            self.boundary_val,
            new_threshold,
        )
        # Decode to features
        new_features = (
            spike_times_to_features(new_spike_times.unsqueeze(1), self.t_target)
            .numpy()
            .ravel()
        )

        # Swap into cached val features
        X_mod = self.baseline_X_val.copy()
        X_mod[:, neuron_idx] = new_features

        y_pred = self.baseline_clf.predict(X_mod)
        return compute_metrics(self.y_val, y_pred)["accuracy"]

    def per_neuron_optimize(
        self,
        initial_thresholds: np.ndarray,
        n_candidates: int = 15,
        search_range: tuple[float, float] = (0.75, 1.25),
    ) -> dict:
        """Optimize each neuron's threshold independently.

        For each neuron, tests n_candidates threshold values while keeping
        all other thresholds at their trained values. Picks the best.

        The per-neuron optimal thresholds are the prediction targets.
        Combined accuracy (applying all changes at once) is also reported
        to assess whether independent optimization transfers.

        Returns dict with per_neuron and combined results.
        """
        num_neurons = len(initial_thresholds)
        baseline_acc = self.evaluate_thresholds(initial_thresholds)

        optimal = initial_thresholds.copy()
        per_neuron_best_acc = np.full(num_neurons, baseline_acc)

        pbar = tqdm(range(num_neurons), desc="Per-neuron optimization", leave=False)
        for neuron_idx in pbar:
            current_val = initial_thresholds[neuron_idx]
            lo = max(current_val * search_range[0], 0.1)
            hi = current_val * search_range[1]
            candidates = np.linspace(lo, hi, n_candidates)

            best_val = current_val
            best_acc = baseline_acc

            for candidate in candidates:
                acc = self._eval_single_neuron_swap(neuron_idx, candidate)
                if acc > best_acc:
                    best_val = candidate
                    best_acc = acc
                elif acc == best_acc and abs(candidate - current_val) < abs(
                    best_val - current_val
                ):
                    # Tie-break: prefer threshold closest to original
                    best_val = candidate

            optimal[neuron_idx] = best_val
            per_neuron_best_acc[neuron_idx] = best_acc
            pbar.set_postfix_str(
                f"neuron={neuron_idx} best={best_acc:.4f} "
                f"delta={best_val - current_val:+.2f}"
            )

        pbar.close()

        deltas = optimal - initial_thresholds
        n_changed = np.sum(np.abs(deltas) > 1e-6)
        n_improved = np.sum(per_neuron_best_acc > baseline_acc + 1e-6)
        avg_improvement = np.mean(per_neuron_best_acc) - baseline_acc

        # Combined accuracy: apply all optimal thresholds together (refit SVC)
        combined_acc = self.evaluate_thresholds_with_refit(optimal)

        # Conservative combined: only apply changes that individually improve > 0.1%
        min_improvement = 0.001
        conservative = initial_thresholds.copy()
        for i in range(num_neurons):
            if per_neuron_best_acc[i] > baseline_acc + min_improvement:
                conservative[i] = optimal[i]
        conservative_acc = self.evaluate_thresholds_with_refit(conservative)
        n_conservative = np.sum(np.abs(conservative - initial_thresholds) > 1e-6)

        tqdm.write(
            f"  Per-neuron: baseline={baseline_acc:.4f} "
            f"avg_per_neuron_best={np.mean(per_neuron_best_acc):.4f} "
            f"(+{avg_improvement:+.4f})"
        )
        tqdm.write(
            f"  Combined (all {n_changed}): {combined_acc:.4f} | "
            f"Conservative ({n_conservative} neurons): {conservative_acc:.4f}"
        )

        return {
            "optimal_thresholds": optimal,
            "baseline_accuracy": baseline_acc,
            "combined_accuracy": combined_acc,
            "conservative_accuracy": conservative_acc,
            "per_neuron_best_acc": per_neuron_best_acc,
            "per_neuron_deltas": deltas,
            "avg_per_neuron_improvement": avg_improvement,
            "n_changed": int(n_changed),
            "n_improved": int(n_improved),
            "n_conservative": int(n_conservative),
        }
