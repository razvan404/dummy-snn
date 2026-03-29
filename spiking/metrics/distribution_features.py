import numpy as np
import torch
from scipy import stats as sp_stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from spiking.layers.integrate_and_fire import IntegrateAndFireLayer


@torch.no_grad()
def compute_vmax_batch(
    layer: IntegrateAndFireLayer,
    input_times_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute peak membrane potentials and spike times for a batch.

    Uses the layer's analytical inference which accumulates potentials
    without firing/reset — the cumulative potential IS V_max.

    Args:
        layer: Trained IF layer with weights.
        input_times_batch: (B, num_inputs) input spike times.

    Returns:
        (V_max, spike_times) both shape (B, num_outputs).
    """
    spike_times, cum_potential = layer.infer_spike_times_and_potentials_batch(
        input_times_batch
    )
    return cum_potential, spike_times


@torch.no_grad()
def compute_vmax_dataset(
    layer: IntegrateAndFireLayer,
    dataloader: DataLoader,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute V_max for an entire dataset with batched inference.

    Args:
        layer: Trained IF layer.
        dataloader: Dataset loader (batch_size=None expected).
        batch_size: Number of samples per inference batch.

    Returns:
        (V_max, spike_times, labels) as numpy arrays.
        V_max: (N, num_outputs), spike_times: (N, num_outputs), labels: (N,).
    """
    layer.eval()
    batched = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False)

    vmax_parts = []
    spike_parts = []
    label_parts = []

    for batch_times, batch_labels in tqdm(batched, desc="Computing V_max", leave=False):
        flat_times = batch_times.flatten(1)
        vmax, spike_times = compute_vmax_batch(layer, flat_times)
        vmax_parts.append(vmax.numpy())
        spike_parts.append(spike_times.numpy())
        label_parts.append(batch_labels.numpy())

    return (
        np.concatenate(vmax_parts, axis=0),
        np.concatenate(spike_parts, axis=0),
        np.concatenate(label_parts, axis=0),
    )


def compute_distribution_features(
    V_max: np.ndarray,
    thresholds: np.ndarray,
    n_bins: int = 16,
) -> dict[str, np.ndarray]:
    """Compute per-neuron distributional features from V_max matrix.

    Args:
        V_max: (N, num_neurons) peak membrane potentials.
        thresholds: (num_neurons,) current threshold values.
        n_bins: Number of histogram bins for entropy computation.

    Returns:
        Dict mapping feature_name -> array of shape (num_neurons,).
    """
    N, num_neurons = V_max.shape
    features = {}

    features["v_mean"] = V_max.mean(axis=0)
    features["v_std"] = V_max.std(axis=0)
    features["v_median"] = np.median(V_max, axis=0)
    features["v_skewness"] = sp_stats.skew(V_max, axis=0)
    features["v_kurtosis"] = sp_stats.kurtosis(V_max, axis=0)

    features["v_quantile_99"] = np.quantile(V_max, 0.99, axis=0)
    features["v_quantile_995"] = np.quantile(V_max, 0.995, axis=0)
    features["v_quantile_998"] = np.quantile(V_max, 0.998, axis=0)

    # Bimodality coefficient: (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    skew = features["v_skewness"]
    kurt = features["v_kurtosis"]  # excess kurtosis from scipy
    if N > 3:
        denom = kurt + 3.0 * (N - 1) ** 2 / ((N - 2) * (N - 3))
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        features["v_bimodality"] = (skew**2 + 1.0) / denom
    else:
        features["v_bimodality"] = np.zeros(num_neurons)

    # Firing ratio: fraction of samples where V_max >= threshold
    features["v_firing_ratio"] = (V_max >= thresholds[np.newaxis, :]).mean(axis=0)

    # Entropy of binned V_max distribution
    entropy = np.zeros(num_neurons)
    for i in range(num_neurons):
        col = V_max[:, i]
        col_range = col.max() - col.min()
        if col_range < 1e-10:
            entropy[i] = 0.0
            continue
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy[i] = -np.sum(probs * np.log(probs))
    features["v_entropy_binned"] = entropy

    return features
