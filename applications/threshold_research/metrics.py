import numpy as np
import torch
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


def compute_classifier_metrics(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Compute per-neuron classifier-aware metrics.

    Returns dict with per-neuron arrays:
      coef_magnitude: mean |W[:, i]| across classes
      misclassified_margin_contribution: avg margin contribution on misclassified samples
      fisher_discriminant_ratio: between-class / within-class variance per feature
      max_feature_correlation: max |pearson_r| with any other feature
    """
    num_features = X_train.shape[1]
    coefs = classifier.coef_  # (num_classes, num_features)

    # coef_magnitude: same as compute_feature_importance
    coef_magnitude = np.mean(np.abs(coefs), axis=0)

    # misclassified_margin_contribution
    y_pred = classifier.predict(X_val)
    misclassified = y_pred != y_val
    margin_contribution = np.zeros(num_features)

    if np.any(misclassified):
        X_mis = X_val[misclassified]
        y_true_mis = y_val[misclassified]
        y_pred_mis = y_pred[misclassified]

        for s in range(len(X_mis)):
            t = y_true_mis[s]
            p = y_pred_mis[s]
            # Positive means neuron pushes toward correct class
            margin_contribution += X_mis[s] * (coefs[t] - coefs[p])

        margin_contribution /= len(X_mis)

    # fisher_discriminant_ratio
    classes = np.unique(y_train)
    overall_mean = np.mean(X_train, axis=0)
    between_var = np.zeros(num_features)
    within_var = np.zeros(num_features)

    for c in classes:
        mask = y_train == c
        class_mean = np.mean(X_train[mask], axis=0)
        n_c = np.sum(mask)
        between_var += n_c * (class_mean - overall_mean) ** 2
        within_var += np.sum((X_train[mask] - class_mean) ** 2, axis=0)

    # When within_var is 0 but between_var > 0, discriminability is perfect
    fisher_discriminant_ratio = np.where(
        within_var > 0,
        between_var / within_var,
        np.where(between_var > 0, np.inf, 0.0),
    )

    # max_feature_correlation (vectorized)
    corr_matrix = np.corrcoef(X_train.T)  # (num_features, num_features)
    np.fill_diagonal(corr_matrix, 0)
    max_feature_correlation = np.max(np.abs(corr_matrix), axis=1)

    return {
        "coef_magnitude": coef_magnitude,
        "misclassified_margin_contribution": margin_contribution,
        "fisher_discriminant_ratio": fisher_discriminant_ratio,
        "max_feature_correlation": max_feature_correlation,
    }
