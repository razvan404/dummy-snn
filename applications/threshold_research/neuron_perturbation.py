import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from torch.utils.data import DataLoader

from applications.common import set_seed
from spiking import load_model
from spiking.evaluation.feature_extraction import (
    spike_times_to_features,
    extract_features,
)
from spiking.evaluation.eval_utils import compute_metrics
from spiking.layers import SpikingSequential


def precompute_cumulative_potentials(
    input_times: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cumulative membrane potentials at each unique input time boundary.

    input_times: (B, I) — spike times for each input, inf means no spike.
    weights: (O, I) — layer weights.

    Returns:
        cum_potentials: (B, O, G) — cumulative potential at each boundary time.
        boundary_times: (G,) — sorted unique finite input times.
    """
    B, I = input_times.shape
    O = weights.shape[0]

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return torch.zeros((B, O, 0)), torch.zeros(0)

    boundary_times = input_times[finite_mask].unique().sort()[0]
    G = len(boundary_times)

    cum_potentials = torch.zeros((B, O, G), dtype=input_times.dtype)
    running = torch.zeros((B, O), dtype=input_times.dtype)

    for g, t in enumerate(boundary_times):
        active = (input_times == t).float()  # (B, I)
        contrib = active @ weights.T  # (B, O)
        running = running + contrib
        cum_potentials[:, :, g] = running

    return cum_potentials, boundary_times


def spike_times_from_potentials(
    cum_potentials: torch.Tensor,
    boundary_times: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Find first threshold crossing time from precomputed cumulative potentials.

    cum_potentials: (B, G) — cumulative potential for one neuron across boundary times.
    boundary_times: (G,) — sorted unique times.
    threshold: scalar threshold value.

    Returns: (B,) — spike time for each sample (inf if no crossing).
    """
    B = cum_potentials.shape[0]
    if cum_potentials.shape[1] == 0:
        return torch.full((B,), float("inf"), dtype=cum_potentials.dtype)

    crossed = cum_potentials >= threshold  # (B, G)
    any_crossed = crossed.any(dim=1)  # (B,)
    first_crossing = crossed.float().argmax(dim=1)  # (B,)

    result = torch.full((B,), float("inf"), dtype=cum_potentials.dtype)
    result[any_crossed] = boundary_times[first_crossing[any_crossed]]

    return result


def run_perturbation_sweep(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    layer_idx: int = 0,
    t_target: float | None = None,
    seed: int = 42,
) -> dict:
    """Run per-neuron threshold perturbation sweep.

    For each neuron, perturbs its threshold by fractions from -0.5 to +0.25 (step 0.025),
    recomputes only that neuron's features, and measures accuracy impact.

    Returns dict with baseline metrics, perturbation matrices, and optimal thresholds.
    """
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    sub_model = sub_model.cpu()

    num_outputs = layer.num_outputs
    original_thresholds = layer.thresholds.detach().clone()

    perturbation_fractions = [round(-0.5 + i * 0.025, 3) for i in range(31)]

    # Extract unperturbed features as baseline
    X_train, y_train = extract_features(sub_model, train_loader, spike_shape, t_target)
    X_val, y_val = extract_features(sub_model, val_loader, spike_shape, t_target)

    # Baseline uses same classifier type as perturbation loop
    baseline_clf = RidgeClassifier()
    baseline_clf.fit(X_train, y_train)
    baseline_metrics = compute_metrics(y_val, baseline_clf.predict(X_val))

    # Precompute cumulative potentials for both train and val
    weights = layer.weights.detach()

    def _collect_input_times(loader):
        batched = DataLoader(loader.dataset, batch_size=256, shuffle=False)
        parts = []
        for batch_times, _labels in batched:
            parts.append(batch_times.flatten(1))
        return torch.cat(parts, dim=0)

    train_input_times = _collect_input_times(train_loader)
    val_input_times = _collect_input_times(val_loader)

    train_cum, train_boundary = precompute_cumulative_potentials(
        train_input_times, weights
    )
    val_cum, val_boundary = precompute_cumulative_potentials(
        val_input_times, weights
    )

    # For each perturbation fraction, retrain classifier per neuron
    accuracy_matrix = np.zeros((num_outputs, len(perturbation_fractions)))
    f1_matrix = np.zeros((num_outputs, len(perturbation_fractions)))

    for frac_idx, frac in enumerate(perturbation_fractions):
        new_thresholds = original_thresholds * (1.0 + frac)

        # Compute perturbed spike times for all neurons (train and val)
        def _perturbed_features(cum_potentials, boundary_times, n_samples):
            spike_times = torch.full(
                (n_samples, num_outputs), float("inf"), dtype=cum_potentials.dtype
            )
            for neuron_idx in range(num_outputs):
                spike_times[:, neuron_idx] = spike_times_from_potentials(
                    cum_potentials[:, neuron_idx, :],
                    boundary_times,
                    new_thresholds[neuron_idx].item(),
                )
            return spike_times_to_features(spike_times, t_target).numpy()

        train_perturbed = _perturbed_features(
            train_cum, train_boundary, train_input_times.shape[0]
        )
        val_perturbed = _perturbed_features(
            val_cum, val_boundary, val_input_times.shape[0]
        )

        # For each neuron, swap that column and retrain
        for neuron_idx in range(num_outputs):
            X_train_mod = X_train.copy()
            X_train_mod[:, neuron_idx] = train_perturbed[:, neuron_idx]
            X_val_mod = X_val.copy()
            X_val_mod[:, neuron_idx] = val_perturbed[:, neuron_idx]

            clf = RidgeClassifier()
            clf.fit(X_train_mod, y_train)
            y_pred = clf.predict(X_val_mod)
            metrics = compute_metrics(y_val, y_pred)
            accuracy_matrix[neuron_idx, frac_idx] = metrics["accuracy"]
            f1_matrix[neuron_idx, frac_idx] = metrics["f1"]

    # Find per-neuron optimal perturbation
    best_frac_indices = accuracy_matrix.argmax(axis=1)
    optimal_fracs = [perturbation_fractions[i] for i in best_frac_indices]
    optimal_thresholds = [
        (original_thresholds[n].item() * (1.0 + optimal_fracs[n]))
        for n in range(num_outputs)
    ]
    optimal_deltas = [
        optimal_thresholds[n] - original_thresholds[n].item()
        for n in range(num_outputs)
    ]

    return {
        "baseline": baseline_metrics,
        "original_thresholds": original_thresholds.tolist(),
        "perturbation_fractions": perturbation_fractions,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "f1_matrix": f1_matrix.tolist(),
        "optimal_thresholds": optimal_thresholds,
        "optimal_deltas": optimal_deltas,
    }
