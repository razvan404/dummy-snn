import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset
from applications.deep_linear.training_plots import save_threshold_distribution
from spiking import load_model, save_model
from spiking.layers import SpikingSequential


def _compute_max_potentials(layer, input_times: torch.Tensor) -> torch.Tensor:
    """Compute the max cumulative membrane potential each neuron reaches.

    Uses the same sorted-weight cumsum as infer_spike_times, but returns
    the final cumulative potential instead of checking threshold crossing.
    Returns a tensor of shape (num_outputs,). If no finite inputs exist,
    returns zeros.
    """
    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return torch.zeros(layer.num_outputs, dtype=input_times.dtype)

    finite_indices = torch.nonzero(finite_mask, as_tuple=True)[0]
    finite_times = input_times[finite_indices]

    _, sort_order = finite_times.sort()
    sorted_indices = finite_indices[sort_order]

    sorted_weights = layer.weights[:, sorted_indices]
    cum_potentials = sorted_weights.cumsum(dim=1)

    return cum_potentials[:, -1]


def percentile_threshold(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    percentile: float,
    layer_idx: int = 0,
    t_target: float | None = None,
):
    """Set thresholds at a percentile of per-neuron max-potential distributions."""
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    preceding = SpikingSequential(*model.layers[:layer_idx]) if layer_idx > 0 else None

    num_neurons = layer.num_outputs
    all_potentials = []

    sub_model.eval()
    with torch.no_grad():
        for times, _label in train_loader:
            input_times = times.flatten()
            if preceding is not None:
                input_times = preceding.infer_spike_times(input_times)
            potentials = _compute_max_potentials(layer, input_times)
            all_potentials.append(potentials)

    # (num_samples, num_neurons) → percentile along sample axis
    potential_matrix = torch.stack(all_potentials, dim=0)
    thresholds = torch.quantile(potential_matrix, percentile / 100.0, dim=0)

    layer.thresholds.data = thresholds.clamp(min=1.0)

    train_m, val_m = evaluate_model(
        sub_model,
        train_loader,
        val_loader,
        spike_shape,
        t_target=t_target,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/model.pth")

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    save_threshold_distribution(
        layer.thresholds, f"{output_dir}/threshold_distribution.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Percentile-based threshold calibration"
    )
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--percentile", type=float, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--t-target", type=float, default=None)
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    percentile_threshold(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        output_dir=args.output_dir,
        percentile=args.percentile,
        layer_idx=args.layer_idx,
        t_target=args.t_target,
    )
