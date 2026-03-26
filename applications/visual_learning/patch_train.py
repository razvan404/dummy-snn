import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from applications.common import set_seed
from applications.datasets.cifar10_patches import Cifar10PatchDataset
from applications.visual_learning.train import (
    DEFAULTS,
    create_stdp,
    create_threshold_adaptation,
    save_weight_figures,
)
from spiking import (
    IntegrateAndFireLayer,
    ConvIntegrateAndFireLayer,
    Learner,
    UnsupervisedTrainer,
    WinnerTakesAll,
    NormalInitialization,
    save_model,
    load_model,
)
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features


@torch.no_grad()
def _evaluate_split(
    conv_layer, encoded_images, labels, pool_size=2, t_target=None, chunk_size=50
):
    """Extract conv features from pre-encoded spike time images in small chunks."""
    import gc

    conv_layer.eval()

    flat_dim = conv_layer.num_filters * pool_size * pool_size
    X = np.empty((len(encoded_images), flat_dim), dtype=np.float32)

    for start in range(0, len(encoded_images), chunk_size):
        end = min(start + chunk_size, len(encoded_images))
        chunk = encoded_images[start:end]

        spike_times = conv_layer.infer_spike_times_batch(chunk)
        features = spike_times_to_features(spike_times, t_target=t_target)
        del spike_times
        pooled = sum_pool_features(features, pool_size)
        del features
        X[start:end] = pooled.flatten(1).numpy()
        del pooled

        if start % 5000 == 0:
            gc.collect()

    return X, labels


def create_fc_layer(
    num_filters: int,
    in_channels: int = 6,
    kernel_size: int = 5,
    params: dict | None = None,
) -> IntegrateAndFireLayer:
    """Create a fully-connected IntegrateAndFireLayer for patch training."""
    p = {**DEFAULTS, **(params or {})}
    num_inputs = in_channels * kernel_size * kernel_size
    init = NormalInitialization(
        avg_threshold=p["threshold_avg"],
        min_threshold=p["threshold_min"],
        std_dev=p["threshold_std"],
    )
    return IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_filters,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )


def create_fc_learner(
    layer: IntegrateAndFireLayer, stdp_variant: str, params: dict | None = None
) -> Learner:
    """Create a Learner with STDP + WTA + threshold adaptation for FC training."""
    stdp = create_stdp(stdp_variant, params)
    adaptation = create_threshold_adaptation(params)
    return Learner(
        layer,
        stdp,
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )


def transfer_weights(
    fc_layer: IntegrateAndFireLayer, conv_layer: ConvIntegrateAndFireLayer
) -> None:
    """Copy FC weights and thresholds to a convolutional layer."""
    conv_layer.weights.data.copy_(
        fc_layer.weights.data.reshape(conv_layer.weights.shape)
    )
    conv_layer.thresholds.data.copy_(fc_layer.thresholds.data)


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
    """Full patch-based training pipeline: patches → FC train → conv eval → save."""
    p = {**DEFAULTS, **(params or {})}
    if num_epochs is None:
        num_epochs = p["num_epochs"]

    set_seed(seed)

    # Load pre-processed dataset
    logger.info("Loading pre-processed CIFAR-10 patches...")
    data = Cifar10PatchDataset(processed_dir)
    num_rounds = min(num_epochs, data.num_rounds)
    logger.info(
        "Patch training: %s STDP, %d filters, %d rounds, seed=%d",
        stdp_variant,
        num_filters,
        num_rounds,
        seed,
    )

    # FC model + learner
    in_channels = data.image_shape[0]
    layer = create_fc_layer(
        num_filters,
        in_channels=in_channels,
        kernel_size=p["kernel_size"],
        params=params,
    )
    learner = create_fc_learner(layer, stdp_variant, params)
    trainer = UnsupervisedTrainer(
        layer,
        learner,
        image_shape=data.image_shape,
        early_stopping=True,
    )

    # Round-based training loop
    for round_idx in range(num_rounds):
        layer.train()
        pbar = tqdm(
            range(data.num_images),
            desc=f"Round {round_idx + 1}/{num_rounds}",
            unit="sample",
        )
        for i in pbar:
            times, _label = data.get_patch(round_idx, i)
            trainer.step_batch(i, times, split="train")
            if i % 200 == 0:
                pbar.set_postfix(sample=i)
        trainer.step_epoch()

    # Free training state before saving/evaluation to avoid OOM
    del trainer, learner
    import gc

    gc.collect()

    # Save FC model immediately after training
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")

    setup_info = {
        "seed": seed,
        "stdp_variant": stdp_variant,
        "num_filters": num_filters,
        "num_rounds": num_rounds,
        "processed_dir": processed_dir,
        **{k: v for k, v in p.items() if k != "num_epochs"},
    }
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)
    logger.info("Model saved to %s/model.pth", output_dir)

    # Transfer weights to conv layer for evaluation
    logger.info("Transferring weights to conv layer...")
    del layer  # free FC layer memory
    conv_layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=p["kernel_size"],
        stride=p["stride"],
        padding=p["padding"],
        threshold_initialization=NormalInitialization(
            avg_threshold=p["threshold_avg"],
            min_threshold=p["threshold_min"],
            std_dev=p["threshold_std"],
        ),
    )
    fc_layer = load_model(f"{output_dir}/model.pth")
    transfer_weights(fc_layer, conv_layer)
    save_weight_figures(conv_layer.weights, f"{output_dir}/weights.png")
    del fc_layer
    gc.collect()

    # Evaluate using pre-encoded images from processed dataset
    logger.info("Evaluating...")
    test_data = torch.load(f"{processed_dir}/test.pt", weights_only=True)

    X_train, y_train = _evaluate_split(
        conv_layer,
        data.images,
        data.labels.numpy(),
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )
    X_test, y_test = _evaluate_split(
        conv_layer,
        test_data["images"],
        test_data["labels"].numpy(),
        pool_size=p["pool_size"],
        t_target=p["target_timestamp"],
    )
    train_m, val_m = evaluate_classifier(X_train, y_train, X_test, y_test)

    logger.info("  Train accuracy: %.4f", train_m["accuracy"])
    logger.info("  Test accuracy:  %.4f", val_m["accuracy"])

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a patch-based SNN on whitened CIFAR-10 (Falez 2020)"
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
            f"logs/patch_learning/{args.stdp}_f{args.num_filters}_s{args.seed}"
        )

    train_model(
        seed=args.seed,
        stdp_variant=args.stdp,
        num_filters=args.num_filters,
        num_epochs=args.num_epochs,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
