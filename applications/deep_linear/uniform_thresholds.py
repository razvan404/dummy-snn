import argparse

import torch
from torch.utils.data import DataLoader

from applications.datasets import create_dataset
from applications.deep_linear.threshold_transform import apply_threshold_transform


def uniform_thresholds(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    layer_idx: int = 0,
    half_width: float = 0.1,
):
    """Replace a layer's thresholds with uniformly distributed values and evaluate."""

    def transform(layer):
        mean_thresh = layer.thresholds.mean().item()
        layer.thresholds.data = (
            torch.empty(layer.thresholds.shape)
            .uniform_(mean_thresh - half_width, mean_thresh + half_width)
            .clamp(min=1.0)
        )

    apply_threshold_transform(
        transform,
        model_path=model_path,
        dataset_loaders=dataset_loaders,
        spike_shape=spike_shape,
        seed=seed,
        output_dir=output_dir,
        layer_idx=layer_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform threshold control")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--half-width", type=float, default=0.1)
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    uniform_thresholds(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        half_width=args.half_width,
    )
