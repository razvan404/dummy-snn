import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from applications.common import set_seed, evaluate_model
from applications.datasets import create_dataset
from applications import default_hyperparams
from applications.deep_linear.model import create_model, ARCHITECTURE
from applications.deep_linear.training_plots import (
    save_winner_counts,
    save_threshold_distribution,
)
from applications.deep_linear.visualize_weights import save_weight_figure
from spiking import (
    Learner,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
    train,
    save_model,
    load_model,
)
from spiking.layers import SpikingSequential


def train_layer(
    *,
    dataset_loaders: tuple[DataLoader, DataLoader],
    spike_shape: tuple[int, ...],
    seed: int,
    avg_threshold: float,
    output_dir: str,
    layer_idx: int = 0,
    model_path: str | None = None,
    num_epochs: int = 30,
    t_objective: float | None = None,
    architecture: list[int] | None = None,
    on_batch_end=None,
    on_epoch_end=None,
):
    """Train a specific layer of a multi-layer SNN and save artifacts.

    For layer_idx > 0, model_path must point to a model with earlier layers trained.
    """
    if architecture is None:
        architecture = ARCHITECTURE
    set_seed(seed)
    train_loader, val_loader = dataset_loaders
    num_inputs = math.prod(spike_shape)

    if model_path is not None:
        model = load_model(model_path)
    else:
        setup = {
            "threshold_init": {
                "avg_threshold": avg_threshold,
                "min_threshold": 1.0,
                "std_dev": 1.0,
            },
        }
        model = create_model(setup, num_inputs, architecture)

    adaptation = CompetitiveThresholdAdaptation(
        **default_hyperparams.THRESHOLD_ADAPTATION
    )
    if t_objective is not None:
        adaptation = SequentialThresholdAdaptation(
            [
                adaptation,
                TargetTimestampAdaptation(
                    target_timestamp=t_objective,
                    **default_hyperparams.THRESHOLD_ADAPTATION,
                ),
            ]
        )

    layer = model.layers[layer_idx]

    learner = Learner(
        layer,
        learning_mechanism=STDP(**default_hyperparams.STDP),
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )

    win_counts = torch.zeros(layer.num_outputs)

    def _tracking_callback(idx, dw, split):
        if split == "train":
            for neuron_idx in learner.neurons_to_learn:
                win_counts[neuron_idx.item()] += 1
        if on_batch_end is not None:
            on_batch_end(idx, dw, split)

    sub_model = SpikingSequential(*model.layers[: layer_idx + 1])

    train(
        sub_model,
        learner,
        train_loader,
        num_epochs,
        image_shape=spike_shape,
        on_batch_end=_tracking_callback,
        on_epoch_end=on_epoch_end,
        progress=False,
    )

    train_m, val_m = evaluate_model(
        sub_model, train_loader, val_loader, spike_shape, t_target=t_objective,
    )

    os.makedirs(output_dir, exist_ok=True)
    save_model(model, f"{output_dir}/model.pth")

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    setup_info = {
        "seed": seed,
        "avg_threshold": avg_threshold,
        "layer_idx": layer_idx,
        "num_epochs": num_epochs,
    }
    if t_objective is not None:
        setup_info["t_objective"] = t_objective
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)

    with open(f"{output_dir}/winner_counts.json", "w") as f:
        json.dump(win_counts.tolist(), f)
    save_winner_counts(win_counts, f"{output_dir}/winner_counts.png")
    save_threshold_distribution(
        layer.thresholds, f"{output_dir}/threshold_distribution.png"
    )

    if layer_idx == 0:
        save_weight_figure(model.layers[0], spike_shape, f"{output_dir}/weights.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a layer of a deep linear SNN")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--avg-threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--t-objective", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=30)
    args = parser.parse_args()

    train_loader, val_loader = create_dataset(args.dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)

    train_layer(
        dataset_loaders=(train_loader, val_loader),
        spike_shape=spike_shape,
        seed=args.seed,
        avg_threshold=args.avg_threshold,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        model_path=args.model_path,
        num_epochs=args.num_epochs,
        t_objective=args.t_objective,
    )
