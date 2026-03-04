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


def weight_threshold(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    layer_idx: int = 0,
    t_target: float | None = None,
):
    """Set thresholds proportional to incoming weight sums and evaluate."""
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    weights = layer.weights.detach()
    weight_sums = weights.sum(dim=1)

    mean_threshold = layer.thresholds.mean().item()
    mean_weight_sum = weight_sums.mean().item()

    # Scale so mean threshold is preserved
    c = mean_threshold / mean_weight_sum if mean_weight_sum != 0 else 1.0
    layer.thresholds.data = (c * weight_sums).clamp(min=1.0)

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
    parser = argparse.ArgumentParser(description="Weight-sum threshold scaling")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--t-target", type=float, default=None)
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    weight_threshold(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        t_target=args.t_target,
    )
