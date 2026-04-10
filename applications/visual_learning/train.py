import argparse
import gc
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from applications.common import set_seed
from applications.default_hyperparams import get_common_hyperparams
from spiking import (
    BiologicalSTDP,
    ConvIntegrateAndFireLayer,
    MultiplicativeSTDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
    NormalInitialization,
    save_model,
)
from spiking.learning.conv_learner import ConvLearner
from spiking.training.conv_trainer import ConvUnsupervisedTrainer
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features

# Paper hyperparameters (Falez 2020 Table I), sourced from dataset-aware defaults
_CIFAR10_PARAMS = get_common_hyperparams("cifar10_whitened")
DEFAULTS = {
    "kernel_size": _CIFAR10_PARAMS["kernel_size"],
    "stride": _CIFAR10_PARAMS["stride"],
    "padding": _CIFAR10_PARAMS["padding"],
    "num_epochs": 100,
    "threshold_avg": _CIFAR10_PARAMS["avg_threshold"],
    "threshold_std": _CIFAR10_PARAMS["std_dev"],
    "threshold_min": _CIFAR10_PARAMS["min_threshold"],
    "threshold_lr": _CIFAR10_PARAMS["threshold_lr"],
    "target_timestamp": _CIFAR10_PARAMS["target_timestamp"],
    "w_min": _CIFAR10_PARAMS["w_min"],
    "w_max": _CIFAR10_PARAMS["w_max"],
    "multiplicative_lr": _CIFAR10_PARAMS["learning_rate"],
    "multiplicative_beta": _CIFAR10_PARAMS["beta"],
    "biological_tau": 0.1,
    "biological_lr": _CIFAR10_PARAMS["learning_rate"],
    "pool_size": _CIFAR10_PARAMS["pool_size"],
    "annealing": _CIFAR10_PARAMS["annealing"],
}


def save_weight_figures(weights: torch.Tensor, output_path: str, ncols: int = 16):
    """Save a grid of learned filter weights as a PNG image.

    :param weights: (num_filters, in_channels, kH, kW) filter weights.
        First 3 channels are positive RGB, next 3 are negative RGB.
    :param output_path: File path for the output PNG.
    :param ncols: Number of columns in the grid.
    """
    num_filters = weights.shape[0]
    nrows = (num_filters + ncols - 1) // ncols
    # Extract positive RGB channels (interleaved: R+, R-, G+, G-, B+, B-)
    rgb = weights[:, [0, 2, 4]].detach().cpu().numpy()
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


def create_stdp(
    variant: str, params: dict | None = None
) -> MultiplicativeSTDP | BiologicalSTDP:
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
        return BiologicalSTDP(
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
                decay_factor=p.get("annealing", 0.95),
            ),
            TargetTimestampAdaptation(
                target_timestamp=p["target_timestamp"],
                min_threshold=p["threshold_min"],
                learning_rate=p["threshold_lr"],
                decay_factor=p.get("annealing", 0.95),
            ),
        ]
    )


def _extract_random_patches(
    images: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Extract one random patch per image as spatial spike times.

    :param images: (N, C, H, W) whitened+encoded spike times.
    :param kernel_size: Patch side length.
    :returns: patches: (N, C, kH, kW) spike times.
    """
    N, C, H, W = images.shape
    max_row = H - kernel_size
    max_col = W - kernel_size
    rows = torch.randint(0, max_row + 1, (N,))
    cols = torch.randint(0, max_col + 1, (N,))

    patches = torch.empty(N, C, kernel_size, kernel_size)
    for i in range(N):
        patches[i] = images[
            i, :, rows[i] : rows[i] + kernel_size, cols[i] : cols[i] + kernel_size
        ]

    return patches


@torch.no_grad()
def _evaluate_split(
    layer: ConvIntegrateAndFireLayer,
    images: torch.Tensor,
    labels: torch.Tensor,
    pool_size: int = 2,
    t_target: float | None = None,
    chunk_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract conv features from pre-encoded spike time images in chunks."""
    layer.eval()
    flat_dim = layer.num_filters * pool_size * pool_size
    X = np.empty((len(images), flat_dim), dtype=np.float32)

    for start in range(0, len(images), chunk_size):
        end = min(start + chunk_size, len(images))
        spike_times = layer.infer_spike_times_batch(images[start:end])
        features = spike_times_to_features(spike_times, t_target=t_target)
        del spike_times
        pooled = sum_pool_features(features, pool_size)
        del features
        X[start:end] = pooled.flatten(1).numpy()
        del pooled

    return X, labels.numpy()


def train_model(
    *,
    seed: int,
    stdp_variant: str,
    num_filters: int,
    num_epochs: int = None,
    processed_dir: str = "data/processed-cifar10",
    output_dir: str,
    params: dict | None = None,
):
    """Paper-aligned patch-based training pipeline.

    Loads pre-whitened+encoded images, extracts one random patch per image
    each epoch (on the fly, matching Falez 2020), trains FC layer with STDP,
    then evaluates via conv feature extraction + LinearSVC.
    """
    p = {**DEFAULTS, **(params or {})}
    if num_epochs is None:
        num_epochs = p["num_epochs"]

    set_seed(seed)
    logger.info(
        "Training: %s STDP, %d filters, %d epochs, seed=%d",
        stdp_variant,
        num_filters,
        num_epochs,
        seed,
    )

    # Load pre-processed dataset (whitened + spike-encoded)
    logger.info("Loading pre-processed data from %s...", processed_dir)
    train_data = torch.load(f"{processed_dir}/train.pt", weights_only=True)
    all_images = train_data["images"]  # (N, 6, 32, 32) spike times
    all_labels = train_data["labels"]  # (N,)
    N = len(all_images)
    in_channels = all_images.shape[1]  # 6
    ksize = p["kernel_size"]
    logger.info("  %d images, %d channels, kernel=%d", N, in_channels, ksize)

    # Conv layer — inherits from IntegrateAndFireLayer, so patch training works
    # directly. Same layer is used for both training and conv evaluation.
    init = NormalInitialization(
        avg_threshold=p["threshold_avg"],
        min_threshold=p["threshold_min"],
        std_dev=p["threshold_std"],
    )
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=ksize,
        stride=p["stride"],
        padding=p["padding"],

        threshold_initialization=init,
        refractory_period=float("inf"),
    )
    # Paper weight init: U(0, 1) per Falez 2019 Table I / Falez 2020 Table I
    torch.nn.init.uniform_(layer.weights, a=p["w_min"], b=p["w_max"])

    stdp = create_stdp(stdp_variant, params)
    adaptation = create_threshold_adaptation(params)
    learner = ConvLearner(
        layer,
        stdp,
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )
    trainer = ConvUnsupervisedTrainer(
        layer,
        learner,
        image_shape=(in_channels, ksize, ksize),
        early_stopping=True,
    )

    # Paper-aligned training: random patches on the fly each epoch
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        patches = _extract_random_patches(all_images, ksize)
        perm = torch.randperm(N)
        layer.train()
        it = tqdm(range(N), desc=f"epoch {epoch}", unit="patch", leave=False)
        for i in it:
            trainer.step_batch(i, patches[perm[i]])
        trainer.step_epoch()

    # Same layer used directly for conv evaluation — no weight copying needed
    logger.info("Evaluating (conv mode)...")
    del learner, trainer, patches
    gc.collect()

    # Evaluate on train and test sets
    X_train, y_train = _evaluate_split(
        layer,
        all_images,
        all_labels,
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )

    test_data = torch.load(f"{processed_dir}/test.pt", weights_only=True)
    X_test, y_test = _evaluate_split(
        layer,
        test_data["images"],
        test_data["labels"],
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )
    del test_data
    gc.collect()

    train_m, val_m = evaluate_classifier(X_train, y_train, X_test, y_test)
    logger.info("  Train accuracy: %.4f", train_m["accuracy"])
    logger.info("  Test accuracy:  %.4f", val_m["accuracy"])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")
    save_weight_figures(layer.weights_4d, f"{output_dir}/weights.png")

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    setup_info = {
        "seed": seed,
        "stdp_variant": stdp_variant,
        "num_filters": num_filters,
        "num_epochs": num_epochs,
        "processed_dir": processed_dir,
        "training_mode": "patch_based_on_the_fly",
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
    parser.add_argument("--processed-dir", type=str, default="data/processed-cifar10")
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
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
