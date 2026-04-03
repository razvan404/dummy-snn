import argparse
import gc
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from applications.common import set_seed
from applications.default_hyperparams import (
    get_common_hyperparams,
    STDP,
    THRESHOLD_INIT,
    THRESHOLD_ADAPTATION,
)
from spiking import (
    BiologicalSTDP,
    ConvIntegrateAndFireLayer,
    Learner,
    MultiplicativeSTDP,
    NormalInitialization,
    UnsupervisedTrainer,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
    save_model,
)
from spiking.evaluation import evaluate_classifier
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features

logger = logging.getLogger(__name__)


def _save_filter_grid(weights_4d: torch.Tensor, path: str, ncols: int = 16):
    """Save learned filter weights as a grid image.

    Works for any number of input channels (2 for DoG, 6 for whitened).
    For 2-channel DoG: shows ON channel as green, OFF as red.
    For 6-channel whitened: shows RGB+ channels.
    """
    import matplotlib.pyplot as plt

    num_filters, C, kH, kW = weights_4d.shape
    nrows = (num_filters + ncols - 1) // ncols
    w = weights_4d.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i >= num_filters:
            continue
        filt = w[i]  # (C, kH, kW)
        if C == 2:
            # DoG: ON=green, OFF=red
            on = filt[0]
            off = filt[1]
            rgb = np.stack([off, on, np.zeros_like(on)], axis=-1)
        elif C >= 6:
            # Whitened: use R+, G+, B+ channels (indices 0, 2, 4)
            rgb = filt[[0, 2, 4]].transpose(1, 2, 0)
        else:
            # Fallback: grayscale average
            rgb = np.stack([filt.mean(0)] * 3, axis=-1)
        fmin, fmax = rgb.min(), rgb.max()
        if fmax > fmin:
            rgb = (rgb - fmin) / (fmax - fmin)
        ax.imshow(rgb, interpolation="nearest")

    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_stdp(variant: str, params: dict) -> MultiplicativeSTDP | BiologicalSTDP:
    """Create STDP learning mechanism."""
    stdp_lr = params.get("stdp_lr", 0.1)
    if variant == "multiplicative":
        return MultiplicativeSTDP(
            learning_rate=stdp_lr,
            decay_factor=params["annealing"],
            beta=params.get("beta", 1.0),
            w_min=params.get("w_min", 0.0),
            w_max=params.get("w_max", 1.0),
        )
    elif variant == "biological":
        return BiologicalSTDP(
            tau_pre=0.1,
            tau_post=0.1,
            max_pre_spike_time=1.0,
            learning_rate=stdp_lr,
            decay_factor=params["annealing"],
            weights_interval=(params.get("w_min", 0.0), params.get("w_max", 1.0)),
        )
    raise ValueError(f"Unknown STDP variant: {variant!r}")


def _create_threshold_adaptation(params: dict) -> SequentialThresholdAdaptation:
    """Create threshold adaptation (competitive + target timestamp)."""
    return SequentialThresholdAdaptation(
        [
            CompetitiveThresholdAdaptation(
                min_threshold=params["min_threshold"],
                learning_rate=params["threshold_lr"],
                decay_factor=params["annealing"],
            ),
            TargetTimestampAdaptation(
                target_timestamp=params["target_timestamp"],
                min_threshold=params["min_threshold"],
                learning_rate=params["threshold_lr"],
                decay_factor=params["annealing"],
            ),
        ]
    )


@torch.no_grad()
def _evaluate_split(
    layer: ConvIntegrateAndFireLayer,
    all_times: torch.Tensor,
    labels: torch.Tensor,
    pool_size: int = 2,
    t_target: float | None = None,
    chunk_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract conv features from full images with pooling.

    The layer was trained on flat patches but evaluates on full spatial images
    via the conv2d path — same weights, different inference mode.
    """
    layer.eval()
    layer.to(device)
    flat_dim = layer.num_filters * pool_size * pool_size
    X = np.empty((len(all_times), flat_dim), dtype=np.float32)

    for start in range(0, len(all_times), chunk_size):
        end = min(start + chunk_size, len(all_times))
        chunk = all_times[start:end].to(device)
        spike_times = layer.infer_spike_times_batch(chunk)
        features = spike_times_to_features(spike_times.cpu(), t_target=t_target)
        pooled = sum_pool_features(features, pool_size)
        X[start:end] = pooled.flatten(1).numpy()
        del spike_times, features, pooled, chunk

    return X, labels.numpy()


def train_conv(
    *,
    dataset: str,
    seed: int = 42,
    stdp_variant: str = "biological",
    num_filters: int = 32,
    kernel_size: int = 5,
    num_patches: int = 50,
    num_epochs: int = 20,
    pool_size: int = 2,
    t_obj: float | None = None,
    device: str = "cpu",
    output_dir: str | None = None,
):
    """Unified patch-based conv SNN training.

    1. Load dataset with patch extraction (training only)
    2. Train ConvIntegrateAndFireLayer on flat patches via STDP
    3. Evaluate on full images via conv inference + sum pooling + Ridge

    The same ConvIntegrateAndFireLayer handles both:
    - Training: flat (C*kH*kW,) patches → FC matmul path (IS-A IntegrateAndFireLayer)
    - Evaluation: (B, C, H, W) full images → conv2d path
    """
    set_seed(seed)
    # Merge global defaults with dataset-specific params
    hp_key = "mnist" if dataset == "mnist_subset" else dataset
    params = {
        **STDP,
        **THRESHOLD_INIT,
        **THRESHOLD_ADAPTATION,
        **get_common_hyperparams(hp_key),
    }
    in_channels = params["in_channels"]
    if t_obj is not None:
        params["target_timestamp"] = t_obj
    t_target = params["target_timestamp"]

    if output_dir is None:
        output_dir = (
            f"logs/conv_learning/{dataset}_{stdp_variant}_f{num_filters}_s{seed}"
        )

    logger.info(
        "Training: %s, %s STDP, %d filters, k=%d, %d patches/img, %d epochs",
        dataset,
        stdp_variant,
        num_filters,
        kernel_size,
        num_patches,
        num_epochs,
    )

    # --- Load datasets ---
    # Training: with patches (each image yields num_patches flat patches)
    # Validation: full images (for conv evaluation)
    from applications.datasets import create_dataset

    train_loader, val_loader = create_dataset(dataset)

    # Get full training data as tensors
    from torch.utils.data import DataLoader

    full_train = DataLoader(
        train_loader.dataset, batch_size=len(train_loader.dataset), shuffle=False
    )
    all_train_times, all_train_labels = next(iter(full_train))

    full_val = DataLoader(
        val_loader.dataset, batch_size=len(val_loader.dataset), shuffle=False
    )
    all_val_times, all_val_labels = next(iter(full_val))

    N = len(all_train_times)
    logger.info("  %d train images, %d val images", N, len(all_val_times))
    logger.info("  Encoded shape: %s", all_train_times.shape)

    # Generate patch positions for training
    from applications.datasets.base import SpikeEncodingDataset

    _, C, H, W = all_train_times.shape
    torch.manual_seed(seed)
    patch_positions = SpikeEncodingDataset._generate_patch_positions(
        N, H, W, kernel_size, num_patches
    )
    logger.info(
        "  Patches: %d per image, %dx%d, %d total",
        num_patches,
        kernel_size,
        kernel_size,
        N * num_patches,
    )

    # --- Create model ---
    num_inputs = in_channels * kernel_size * kernel_size
    init = NormalInitialization(
        avg_threshold=params["avg_threshold"],
        min_threshold=params["min_threshold"],
        std_dev=params["std_dev"],
    )
    layer = ConvIntegrateAndFireLayer(
        in_channels=in_channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )
    w_min = params.get("w_min", 0.0)
    w_max = params.get("w_max", 1.0)
    torch.nn.init.normal_(layer.weights, mean=(w_min + w_max) / 2, std=0.01)

    stdp = _create_stdp(stdp_variant, params)
    adaptation = _create_threshold_adaptation(params)
    learner = Learner(
        layer, stdp, competition=WinnerTakesAll(), threshold_adaptation=adaptation
    )
    # Training is always on CPU (per-patch STDP is serial, GPU adds overhead)
    trainer = UnsupervisedTrainer(
        layer,
        learner,
        image_shape=(num_inputs,),
        early_stopping=True,
        device="cpu",
    )

    # --- Train on patches ---
    total_patches = N * num_patches
    for epoch in range(num_epochs):
        layer.train()
        perm = torch.randperm(N)
        patch_count = 0
        pbar = tqdm(
            total=total_patches, desc=f"Epoch {epoch+1}/{num_epochs}", unit="patch"
        )
        for img_idx in perm:
            positions = patch_positions[img_idx]  # (num_patches, 2)
            times = all_train_times[img_idx]  # (C, H, W)
            for r, c in positions:
                patch = times[:, r : r + kernel_size, c : c + kernel_size]
                trainer.step_batch(patch_count, patch.flatten())
                patch_count += 1
                pbar.update(1)
        pbar.close()
        trainer.step_epoch()

    # --- Evaluate on full images ---
    # Use GPU for evaluation if available (batched conv2d benefits from GPU)
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Evaluating (conv mode with %dx%d pooling, device=%s)...",
        pool_size, pool_size, eval_device,
    )
    del learner, trainer
    gc.collect()

    X_train, y_train = _evaluate_split(
        layer,
        all_train_times,
        all_train_labels,
        pool_size=pool_size,
        t_target=t_target,
        device=eval_device,
    )
    X_val, y_val = _evaluate_split(
        layer,
        all_val_times,
        all_val_labels,
        pool_size=pool_size,
        t_target=t_target,
        device=eval_device,
    )

    train_m, val_m = evaluate_classifier(X_train, y_train, X_val, y_val)
    logger.info("  Train accuracy: %.4f", train_m["accuracy"])
    logger.info("  Val accuracy:   %.4f", val_m["accuracy"])

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    save_model(layer, f"{output_dir}/model.pth")

    # Save weight visualization
    _save_filter_grid(
        layer.weights_4d, f"{output_dir}/weights.png", ncols=min(16, num_filters)
    )

    metrics = {"train": train_m, "validation": val_m}
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    setup_info = {
        "dataset": dataset,
        "seed": seed,
        "stdp_variant": stdp_variant,
        "num_filters": num_filters,
        "kernel_size": kernel_size,
        "num_patches": num_patches,
        "num_epochs": num_epochs,
        "pool_size": pool_size,
        "training_mode": "patch_based_cached",
        **params,
    }
    with open(f"{output_dir}/setup.json", "w") as f:
        json.dump(setup_info, f, indent=4)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Unified conv SNN training with patch-based learning"
    )
    parser.add_argument(
        "dataset", type=str, help="Dataset name (mnist, fashion_mnist, cifar10)"
    )
    parser.add_argument(
        "--stdp",
        type=str,
        default="biological",
        choices=["multiplicative", "biological"],
    )
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--num-patches", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument(
        "--t-obj", type=float, default=None, help="Target spike time override"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    train_conv(
        dataset=args.dataset,
        seed=args.seed,
        stdp_variant=args.stdp,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        num_patches=args.num_patches,
        num_epochs=args.num_epochs,
        pool_size=args.pool_size,
        t_obj=args.t_obj,
        device=args.device,
        output_dir=args.output_dir,
    )
