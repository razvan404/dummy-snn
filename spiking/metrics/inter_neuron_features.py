import numpy as np


def compute_inter_neuron_features(
    weights: np.ndarray,
    V_max: np.ndarray,
    spike_times: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute inter-neuron competition features.

    Args:
        weights: (num_neurons, num_inputs) weight matrix.
        V_max: (N, num_neurons) peak membrane potentials.
        spike_times: (N, num_neurons) output spike times (inf = no spike).
        thresholds: (num_neurons,) threshold values.

    Returns:
        Dict mapping feature_name -> array of shape (num_neurons,).
    """
    num_neurons = weights.shape[0]
    N = V_max.shape[0]

    # --- Weight overlap (cosine similarity) ---
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    normalized = weights / norms
    cos_sim = normalized @ normalized.T  # (O, O)
    np.fill_diagonal(cos_sim, 0.0)

    features = {}
    features["weight_overlap_mean"] = cos_sim.sum(axis=1) / (num_neurons - 1)
    features["weight_overlap_max"] = cos_sim.max(axis=1)

    # Top-5 nearest neighbors (or all if < 5 other neurons)
    k = min(5, num_neurons - 1)
    if k > 0:
        top_k_indices = np.argpartition(-cos_sim, k, axis=1)[:, :k]
        top_k_vals = np.take_along_axis(cos_sim, top_k_indices, axis=1)
        features["weight_overlap_top5"] = top_k_vals.mean(axis=1)
    else:
        features["weight_overlap_top5"] = np.zeros(num_neurons)

    # --- Competition margins ---
    # For each sample, find which neuron wins (earliest finite spike)
    finite_mask = np.isfinite(spike_times)  # (N, O)
    spike_for_wta = np.where(finite_mask, spike_times, np.inf)
    winners = np.argmin(spike_for_wta, axis=1)  # (N,)
    winner_times = spike_for_wta[np.arange(N), winners]  # (N,)

    # Runner-up: second smallest spike time
    # Set winner's time to inf, find next min
    spike_no_winner = spike_for_wta.copy()
    spike_no_winner[np.arange(N), winners] = np.inf
    runner_up_times = np.min(spike_no_winner, axis=1)  # (N,)

    # Margin = runner_up - winner (larger = more confident win)
    with np.errstate(invalid="ignore"):
        margins = runner_up_times - winner_times  # (N,)

    # Only consider samples where both winner and runner-up spiked
    valid = np.isfinite(winner_times) & np.isfinite(runner_up_times)

    competition_margin_mean = np.zeros(num_neurons)
    competition_margin_std = np.zeros(num_neurons)
    narrow_win_fraction = np.zeros(num_neurons)

    if valid.any():
        valid_margins = margins[valid]
        valid_winners = winners[valid]

        # Narrow win threshold: 10th percentile of all margins
        margin_p10 = np.percentile(valid_margins, 10) if len(valid_margins) > 0 else 0.0

        for i in range(num_neurons):
            neuron_mask = valid_winners == i
            if not neuron_mask.any():
                continue
            neuron_margins = valid_margins[neuron_mask]
            competition_margin_mean[i] = neuron_margins.mean()
            competition_margin_std[i] = (
                neuron_margins.std() if len(neuron_margins) > 1 else 0.0
            )
            narrow_win_fraction[i] = (neuron_margins < margin_p10).mean()

    features["competition_margin_mean"] = competition_margin_mean
    features["competition_margin_std"] = competition_margin_std
    features["narrow_win_fraction"] = narrow_win_fraction

    return features
