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


def _count_fires(model, loader, spike_shape, layer_idx):
    """Run inference and count how many times each neuron in layer fires."""
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
    layer = model.layers[layer_idx]
    num_neurons = layer.num_outputs
    fire_counts = torch.zeros(num_neurons)

    sub_model.eval()
    with torch.no_grad():
        for times, _label in loader:
            spike_times = sub_model.infer_spike_times(times.flatten())
            fire_counts += torch.isfinite(spike_times).float()

    return fire_counts


def activity_threshold(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    layer_idx: int = 0,
    t_target: float | None = None,
):
    """Adjust thresholds based on firing activity and evaluate."""
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    fire_counts = _count_fires(model, train_loader, spike_shape, layer_idx)

    mean_rate = fire_counts.mean()
    # Avoid division by zero for neurons that never fire
    safe_rates = fire_counts.clamp(min=1.0)
    scale = mean_rate / safe_rates
    layer.thresholds.data = (layer.thresholds.data * scale).clamp(min=1.0)

    train_m, val_m = evaluate_model(
        sub_model,
        train_loader,
        val_loader,
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
    parser = argparse.ArgumentParser(description="Activity-based threshold adjustment")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--t-target", type=float, default=None)
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    activity_threshold(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        t_target=args.t_target,
    )
