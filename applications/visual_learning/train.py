import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from applications.common import set_seed, aggregate_metrics
from applications.datasets.cifar10_whitened import create_cifar10_whitened
from spiking import (
    ConvIntegrateAndFireLayer,
    ConvLearner,
    ConvUnsupervisedTrainer,
    MultiplicativeSTDP,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
    NormalInitialization,
    save_model,
)
from spiking.evaluation import extract_conv_features, evaluate_classifier


# Paper hyperparameters (Table I)
DEFAULTS = {
    "kernel_size": 5,
    "stride": 1,
    "padding": 0,
    "num_epochs": 3,
    "threshold_avg": 10.0,
    "threshold_std": 0.1,
    "threshold_min": 1.0,
    "threshold_lr": 1.0,
    "target_timestamp": 0.97,
    "w_min": 0.0,
    "w_max": 1.0,
    "multiplicative_lr": 0.1,
    "multiplicative_beta": 1.0,
    "biological_tau": 0.1,
    "biological_lr": 0.1,
    "pool_size": 2,
    "whitening_patch_size": 9,
    "whitening_epsilon": 1e-2,
    "whitening_rho": 0.15,
    "annealing": 0.95,
}


def save_weight_figures(weights: torch.Tensor, output_path: str, ncols: int = 16):
    """Save a grid of learned filter weights as a PNG image.

    Args:
        weights: (num_filters, in_channels, kH, kW) filter weights.
            First 3 channels are positive RGB, next 3 are negative RGB.
        output_path: File path for the output PNG.
        ncols: Number of columns in the grid.
    """
    num_filters = weights.shape[0]
    nrows = (num_filters + ncols - 1) // ncols
    # Extract positive RGB channels (interleaved: R+, R-, G+, G-, B+, B-)
    rgb = weights[:, [0, 2, 4]].detach().cpu().numpy()
    # Normalize each filter to [0, 1] for display
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i < num_filters:
            filt = rgb[i].transpose(1, 2, 0)  # (kH, kW, 3)
            fmin, fmax = filt.min(), filt.max()
            if fmax > fmin:
                filt = (filt - fmin) / (fmax - fmin)
            ax.imshow(filt, interpolation="nearest")
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_layer(
    num_filters: int, in_channels: int = 6, params: dict | None = None
) -> ConvIntegrateAndFireLayer:
    """Create a ConvIntegrateAndFireLayer with paper defaults."""
    p = {**DEFAULTS, **(params or {})}
    init = NormalInitialization(
        avg_threshold=p["threshold_avg"],
        min_threshold=p["threshold_min"],
        std_dev=p["threshold_std"],
    )
    return ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=p["kernel_size"],
        stride=p["stride"],
        padding=p["padding"],
        threshold_initialization=init,
        refractory_period=float("inf"),
    )


def create_stdp(variant: str, params: dict | None = None) -> MultiplicativeSTDP | STDP:
    """Create STDP learning mechanism by variant name."""
    p = {**DEFAULTS, **(params or {})}
    if variant == "multiplicative":
        return MultiplicativeSTDP(
            learning_rate=p["multiplicative_lr"],
            decay_factor=p["annealing"],
            beta=p["multiplicative_beta"],
            w_min=p["w_min"],
            w_max=p["w_max"],
        )
    elif variant == "biological":
        return STDP(
            tau_pre=p["biological_tau"],
            tau_post=p["biological_tau"],
            max_pre_spike_time=1.0,
            learning_rate=p["biological_lr"],
            decay_factor=p["annealing"],
            weights_interval=(p["w_min"], p["w_max"]),
        )
    else:
        raise ValueError(f"Unknown STDP variant: {variant!r}")


def create_threshold_adaptation(
    params: dict | None = None,
) -> SequentialThresholdAdaptation:
    """Create paper-default threshold adaptation (competitive + target timestamp)."""
    p = {**DEFAULTS, **(params or {})}
    return SequentialThresholdAdaptation(
        [
            CompetitiveThresholdAdaptation(
                min_threshold=p["threshold_min"],
                learning_rate=p["threshold_lr"],
                decay_factor=1.0,
            ),
            TargetTimestampAdaptation(
                target_timestamp=p["target_timestamp"],
                min_threshold=p["threshold_min"],
                learning_rate=p["threshold_lr"],
                decay_factor=1.0,
            ),
        ]
    )


def create_learner(
    layer: ConvIntegrateAndFireLayer, stdp_variant: str, params: dict | None = None
) -> ConvLearner:
    """Create a ConvLearner with paper-default threshold adaptation."""
    stdp = create_stdp(stdp_variant, params)
    adaptation = create_threshold_adaptation(params)
    return ConvLearner(
        layer,
        stdp,
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )


def train_model(
    *,
    seed: int,
    stdp_variant: str,
    num_filters: int,
    num_epochs: int = None,
    output_dir: str,
    params: dict | None = None,
):
    """Full training pipeline: dataset → model → train → evaluate → save."""
    p = {**DEFAULTS, **(params or {})}
    if num_epochs is None:
        num_epochs = p["num_epochs"]

    set_seed(seed)
    logger.info(
        "Training: %s STDP, %d filters, seed=%d", stdp_variant, num_filters, seed
    )

    # Dataset
    logger.info("Loading whitened CIFAR-10...")
    train_loader, val_loader = create_cifar10_whitened(
        patch_size=p["whitening_patch_size"],
        epsilon=p["whitening_epsilon"],
        rho=p["whitening_rho"],
    )
    image_shape = train_loader.dataset.image_shape

    # Model + learner
    layer = create_layer(num_filters, in_channels=image_shape[0], params=params)
    learner = create_learner(layer, stdp_variant, params)
    trainer = ConvUnsupervisedTrainer(
        layer,
        learner,
        image_shape=image_shape,
        early_stopping=True,
    )

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        trainer.step_loader(train_loader, split="train", progress=True)
        trainer.step_epoch()

    # Evaluate
    logger.info("Evaluating...")
    X_train, y_train = extract_conv_features(
        layer,
        train_loader,
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )
    X_test, y_test = extract_conv_features(
        layer,
        val_loader,
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )
    train_m, val_m = evaluate_classifier(X_train, y_train, X_test, y_test)

    logger.info("  Train accuracy: %.4f", train_m["accuracy"])
    logger.info("  Test accuracy:  %.4f", val_m["accuracy"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")
    save_weight_figures(layer.weights, f"{output_dir}/weights.png")

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    setup_info = {
        "seed": seed,
        "stdp_variant": stdp_variant,
        "num_filters": num_filters,
        "num_epochs": num_epochs,
        **{k: v for k, v in p.items() if k != "num_epochs"},
    }
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a conv SNN on whitened CIFAR-10 (Falez 2020)"
    )
    parser.add_argument(
        "--stdp",
        type=str,
        default="multiplicative",
        choices=["multiplicative", "biological"],
        help="STDP variant to use",
    )
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f"logs/visual_learning/{args.stdp}_f{args.num_filters}_s{args.seed}"
        )

    train_model(
        seed=args.seed,
        stdp_variant=args.stdp,
        num_filters=args.num_filters,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
    )
