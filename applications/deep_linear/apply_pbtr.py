import argparse
import json
import os

from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset
from applications.deep_linear.training_plots import save_threshold_distribution
from applications.deep_linear.visualize_weights import save_weight_figure
from spiking import (
    Learner,
    PlasticityBalanceAdaptation,
    train,
    load_model,
    save_model,
)
from spiking.layers import SpikingSequential


def apply_pbtr(
    *,
    model_path: str,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    output_dir: str,
    layer_idx: int = 0,
    num_epochs: int = 10,
    on_batch_end=None,
    t_target: float | None = None,
    sign_only: bool = False,
):
    """Apply PBTR to a specific layer of a trained model and save artifacts."""
    set_seed(seed)
    train_loader, val_loader = dataset_loaders

    model = load_model(model_path)
    layer = model.layers[layer_idx]
    max_threshold = layer.thresholds.mean().item() * 2

    learner = Learner(
        layer,
        learning_mechanism=None,
        threshold_adaptation=PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=max_threshold,
            sign_only=sign_only,
        ),
    )

    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    train(
        sub_model,
        learner,
        train_loader,
        num_epochs,
        image_shape=spike_shape,
        early_stopping=False,
        on_batch_end=on_batch_end,
        progress=False,
    )

    train_m, val_m = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape, t_target=t_target,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/model.pth")

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    save_threshold_distribution(
        layer.thresholds, f"{output_dir}/threshold_distribution.png"
    )

    if layer_idx == 0:
        save_weight_figure(layer, spike_shape, f"{output_dir}/weights.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply PBTR post-training")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--t-target", type=float, default=None)
    parser.add_argument("--sign-only", action="store_true")
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    apply_pbtr(
        model_path=args.model_path,
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        num_epochs=args.num_epochs,
        t_target=args.t_target,
        sign_only=args.sign_only,
    )
