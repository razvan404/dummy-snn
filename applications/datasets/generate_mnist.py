import argparse
import json
import logging
import os

import torch
import torchvision

from spiking.preprocessing import (
    apply_difference_of_gaussians_filter_batch,
    apply_latency_encoding,
    discretize_times,
)

logger = logging.getLogger(__name__)


def generate(
    output_dir: str = "data/processed-mnist",
    root: str = "data",
    patch_size: int = 5,
    patches_per_image: int = 50,
    sigma_center: float = 1.0,
    sigma_surround: float = 4.0,
    num_bins: int = 64,
    seed: int = 42,
):
    """Generate pre-processed MNIST dataset with DoG encoding.

    Saves DoG-encoded+latency spike times, deterministic patch positions,
    and per-round shuffled image orderings to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    logger.info("Loading MNIST...")
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True)

    train_images = train_ds.data.float() / 255.0  # (N, 28, 28)
    train_labels = train_ds.targets.long()
    test_images = test_ds.data.float() / 255.0
    test_labels = test_ds.targets.long()

    # Apply DoG encoding → (N, 2, 28, 28) ON/OFF channels
    logger.info(
        "Applying DoG filter (sigma_center=%.1f, sigma_surround=%.1f)...",
        sigma_center,
        sigma_surround,
    )
    train_dog = apply_difference_of_gaussians_filter_batch(
        train_images, sigma_center=sigma_center, sigma_surround=sigma_surround
    )
    test_dog = apply_difference_of_gaussians_filter_batch(
        test_images, sigma_center=sigma_center, sigma_surround=sigma_surround
    )

    # Latency encoding + discretization
    logger.info("Applying latency encoding (%d bins)...", num_bins)
    train_encoded = discretize_times(apply_latency_encoding(train_dog), num_bins=num_bins)
    test_encoded = discretize_times(apply_latency_encoding(test_dog), num_bins=num_bins)

    # Generate patch positions and round orderings
    N = len(train_images)
    H, W = train_images.shape[1], train_images.shape[2]
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

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(
            {
                "patch_size": patch_size,
                "patches_per_image": patches_per_image,
                "sigma_center": sigma_center,
                "sigma_surround": sigma_surround,
                "num_bins": num_bins,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate pre-processed DoG-encoded MNIST dataset"
    )
    parser.add_argument("--output-dir", default="data/processed-mnist")
    parser.add_argument("--root", default="data")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--patches-per-image", type=int, default=50)
    parser.add_argument("--sigma-center", type=float, default=1.0)
    parser.add_argument("--sigma-surround", type=float, default=4.0)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        root=args.root,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        sigma_center=args.sigma_center,
        sigma_surround=args.sigma_surround,
        num_bins=args.num_bins,
        seed=args.seed,
    )
