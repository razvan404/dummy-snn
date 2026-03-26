import argparse
import json
import logging
import os

import torch
import torchvision

logger = logging.getLogger(__name__)

from spiking.preprocessing.whitening_kernels import (
    fit_whitening_kernels,
    apply_whitening_kernels,
)
from spiking.preprocessing.whitened_spike_encoding import encode_whitened_image
from spiking.preprocessing.latency_encoding import discretize_times


def _load_cifar10_images(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Load raw CIFAR-10 as (N, 3, 32, 32) float [0, 1] and labels."""
    dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)
    images = torch.from_numpy(dataset.data).float() / 255.0
    images = images.permute(0, 3, 1, 2)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    return images, labels


def _encode_all(whitened: torch.Tensor) -> torch.Tensor:
    """Apply per-sample spike encoding + discretization to whitened images."""
    return torch.stack(
        [
            discretize_times(encode_whitened_image(whitened[i]))
            for i in range(len(whitened))
        ]
    )


def generate(
    output_dir: str = "data/processed-cifar10",
    root: str = "data",
    patch_size: int = 5,
    patches_per_image: int = 100,
    whitening_patch_size: int = 9,
    epsilon: float = 1e-2,
    rho: float = 0.15,
    n_whitening_patches: int = 1_000_000,
    seed: int = 42,
):
    """Generate pre-processed CIFAR-10 dataset with deterministic patch ordering.

    Saves whitened+encoded images, whitening kernels, patch positions,
    and per-round shuffled image orderings to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    logger.info("Loading CIFAR-10...")
    train_images, train_labels = _load_cifar10_images(root, train=True)
    test_images, test_labels = _load_cifar10_images(root, train=False)

    # Fit whitening kernels on training data
    logger.info("Fitting whitening kernels (%d patches)...", n_whitening_patches)
    kernels, mean = fit_whitening_kernels(
        train_images,
        patch_size=whitening_patch_size,
        n_patches=n_whitening_patches,
        epsilon=epsilon,
        rho=rho,
    )

    # Whiten and encode both splits
    logger.info("Whitening + encoding %d training images...", len(train_images))
    train_encoded = _encode_all(apply_whitening_kernels(train_images, kernels, mean))
    logger.info("Whitening + encoding %d test images...", len(test_images))
    test_encoded = _encode_all(apply_whitening_kernels(test_images, kernels, mean))

    # Generate patch positions and round orderings
    N = len(train_images)
    H, W = train_images.shape[2], train_images.shape[3]
    max_row = H - patch_size
    max_col = W - patch_size

    logger.info(
        "Generating %d patch positions per image (seed=%d)...", patches_per_image, seed
    )
    rng = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, max_row + 1, (N, patches_per_image), generator=rng)
    cols = torch.randint(0, max_col + 1, (N, patches_per_image), generator=rng)
    patch_positions = torch.stack([rows, cols], dim=-1)  # (N, P, 2)

    logger.info("Generating %d round orderings...", patches_per_image)
    round_orders = torch.stack(
        [torch.randperm(N, generator=rng) for _ in range(patches_per_image)]
    )  # (P, N)

    # Save
    logger.info("Saving to %s/...", output_dir)
    torch.save(
        {
            "images": train_encoded,
            "labels": train_labels,
            "patch_positions": patch_positions,
            "round_orders": round_orders,
        },
        f"{output_dir}/train.pt",
    )

    torch.save(
        {
            "images": test_encoded,
            "labels": test_labels,
        },
        f"{output_dir}/test.pt",
    )

    torch.save(kernels, f"{output_dir}/kernels.pt")
    torch.save(mean, f"{output_dir}/mean.pt")

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(
            {
                "patch_size": patch_size,
                "patches_per_image": patches_per_image,
                "whitening_patch_size": whitening_patch_size,
                "epsilon": epsilon,
                "rho": rho,
                "n_whitening_patches": n_whitening_patches,
                "seed": seed,
            },
            f,
            indent=4,
        )

    logger.info("Done.")
    logger.info(
        "  train.pt: images %s, positions %s, orders %s",
        tuple(train_encoded.shape),
        tuple(patch_positions.shape),
        tuple(round_orders.shape),
    )
    logger.info("  test.pt:  images %s", tuple(test_encoded.shape))
    logger.info("  kernels.pt: %s", tuple(kernels.shape))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate pre-processed whitened CIFAR-10 dataset"
    )
    parser.add_argument("--output-dir", default="data/processed-cifar10")
    parser.add_argument("--root", default="data")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--patches-per-image", type=int, default=100)
    parser.add_argument("--whitening-patch-size", type=int, default=9)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--n-whitening-patches", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        root=args.root,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        whitening_patch_size=args.whitening_patch_size,
        epsilon=args.epsilon,
        rho=args.rho,
        n_whitening_patches=args.n_whitening_patches,
        seed=args.seed,
    )
