"""Precompute per-neuron feature cache for all perturbation levels.

For each neuron and each threshold level, computes pooled features for the
entire dataset. The cache enables instant Woodbury-based coordinate descent
across multiple rounds without re-running SNN inference.

Architecture:
  Phase 1 (per chunk): conv2d accumulation → store potentials at all T steps
  Phase 2 (per neuron): scan precomputed potentials for 21 threshold levels
  Output: cache[neuron, level, image, pool_feature]

Memory: 256 neurons × 21 levels × 60k images × 4 pool × 4 bytes ≈ 5.2 GB
GPU: ~3.3 GB per chunk (potentials) — fits on 20 GB card
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from applications.common import load_split_data, resolve_model_dir, set_seed
from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.utils.checkpoints import load_model

logger = logging.getLogger(__name__)


def compute_feature_cache(
    weights_4d: torch.Tensor,
    original_thresholds: torch.Tensor,
    images: torch.Tensor,
    t_target: float,
    pool_size: int,
    stride: int,
    padding: int,
    perturbation_fractions: list[float],
    device: str = "cuda",
    chunk_size: int = 32,
) -> np.ndarray:
    """Precompute features for all neurons × all perturbation levels.

    Two-phase approach: conv2d once per chunk, threshold check per neuron.

    :param perturbation_fractions: list of fractional perturbations (e.g., [-0.5, -0.45, ..., 0.5]).
    :returns: (num_filters, num_fracs, N, pool_features) float32 array.
    """
    N = len(images)
    num_filters = weights_4d.shape[0]
    kH = weights_4d.shape[2]
    oH = (images.shape[2] + 2 * padding - kH) // stride + 1
    oW = (images.shape[3] + 2 * padding - kH) // stride + 1
    num_fracs = len(perturbation_fractions)
    flat_dim = pool_size * pool_size

    w = weights_4d.to(device)
    orig_thresh = original_thresholds.to(device)

    # Build per-neuron threshold levels: (F, num_fracs)
    frac_tensor = torch.tensor(
        perturbation_fractions, dtype=torch.float32, device=device
    )
    thresh_levels = orig_thresh.unsqueeze(1) * (
        1.0 + frac_tensor.unsqueeze(0)
    )  # (F, num_fracs)

    # Output: (F, num_fracs, N, flat_dim)
    cache = np.zeros((num_filters, num_fracs, N, flat_dim), dtype=np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size
    t0 = time.time()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        inp = images[start:end].to(device)
        B = end - start

        # --- Phase 1: Precompute potentials (conv2d, ONCE) ---
        finite_mask = torch.isfinite(inp)
        if not finite_mask.any():
            continue

        unique_times = inp[finite_mask].unique().sort()[0]
        T = len(unique_times)

        # Store cumulative potential at each time step: (T, B, F, oH, oW)
        all_potentials = torch.empty(
            T,
            B,
            num_filters,
            oH,
            oW,
            dtype=inp.dtype,
            device=device,
        )
        cum = torch.zeros(B, num_filters, oH, oW, dtype=inp.dtype, device=device)
        for k in range(T):
            active = (inp == unique_times[k]).float()
            contrib = F.conv2d(active, w, stride=stride, padding=padding)
            cum = cum + contrib
            all_potentials[k] = cum

        # --- Phase 2: Vectorized threshold check (all neurons at once) ---
        # thresh_levels: (F, num_fracs) on device
        # all_potentials: (T, B, F, oH, oW)
        # result: (F, num_fracs, B, oH, oW)
        result = torch.full(
            (num_filters, num_fracs, B, oH, oW),
            float("inf"),
            dtype=inp.dtype,
            device=device,
        )
        not_yet = torch.ones(
            (num_filters, num_fracs, B, oH, oW),
            dtype=torch.bool,
            device=device,
        )
        # thresh_levels viewed as (F, num_fracs, 1, 1, 1)
        tv = thresh_levels.view(num_filters, num_fracs, 1, 1, 1)

        for k in range(T):
            # pot: (B, F, oH, oW) → (F, 1, B, oH, oW)
            pot = all_potentials[k].permute(1, 0, 2, 3).unsqueeze(1)
            crossed = (pot >= tv) & not_yet
            result[crossed] = unique_times[k]
            not_yet &= ~crossed
            if not not_yet.any():
                break

        del all_potentials, cum, not_yet

        # Convert to features per neuron: (F, num_fracs, B, oH, oW) → cache
        for f_idx in range(num_filters):
            # (num_fracs, B, oH, oW) → (num_fracs*B, 1, oH, oW)
            r = result[f_idx].unsqueeze(2)  # (num_fracs, B, 1, oH, oW)
            flat_4d = r.reshape(num_fracs * B, 1, oH, oW).cpu()
            feat = spike_times_to_features(flat_4d, t_target=t_target)
            pooled = sum_pool_features(feat, pool_size)
            cache[f_idx, :, start:end, :] = (
                pooled.flatten(1).numpy().reshape(num_fracs, B, flat_dim)
            )
        del result

        if (chunk_idx + 1) % 20 == 0 or chunk_idx == n_chunks - 1:
            elapsed = time.time() - t0
            rate = (chunk_idx + 1) / elapsed
            eta = (n_chunks - chunk_idx - 1) / rate if rate > 0 else 0
            logger.info(
                "  chunk %d/%d (%.0fs elapsed, ETA %.0fs)",
                chunk_idx + 1,
                n_chunks,
                elapsed,
                eta,
            )

    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute per-neuron feature cache")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--t-obj", type=float, default=0.7)
    parser.add_argument(
        "--step-size", type=float, default=0.05, help="Perturbation step size"
    )
    parser.add_argument(
        "--max-drift", type=float, default=0.5, help="Max total drift (fraction)"
    )
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    model_dir = resolve_model_dir(args.dataset, args.num_filters, args.t_obj, args.seed)
    model_path = f"{model_dir}/model.pth"
    with open(f"{model_dir}/setup.json") as f:
        t_target = json.load(f).get("target_timestamp", args.t_obj)

    layer = load_model(model_path)
    logger.info("Loaded model from %s", model_path)

    # Build perturbation fractions: -max_drift to +max_drift in step_size increments
    n_steps = int(round(args.max_drift / args.step_size))
    fractions = [round(i * args.step_size, 6) for i in range(-n_steps, n_steps + 1)]
    logger.info(
        "%d perturbation levels: [%.3f, %.3f] step=%.3f",
        len(fractions),
        fractions[0],
        fractions[-1],
        args.step_size,
    )

    # Load data
    logger.info("Loading data...")
    train_data, test_data = load_split_data(args.dataset)
    train_images = train_data["images"]
    test_images = test_data["images"]
    y_train = train_data["labels"].numpy()
    y_test = test_data["labels"].numpy()
    logger.info("Train: %d, Test: %d", len(train_images), len(test_images))

    weights_4d = layer.weights_4d.detach()
    original_thresholds = layer.thresholds.detach().clone()

    # Compute cache for train
    logger.info("Computing train cache...")
    t0 = time.time()
    train_cache = compute_feature_cache(
        weights_4d,
        original_thresholds,
        train_images,
        t_target,
        args.pool_size,
        layer.stride,
        layer.padding,
        fractions,
        args.device,
        args.chunk_size,
    )
    logger.info("Train cache: %.1fs, shape %s", time.time() - t0, train_cache.shape)

    # Compute cache for test
    logger.info("Computing test cache...")
    t0 = time.time()
    test_cache = compute_feature_cache(
        weights_4d,
        original_thresholds,
        test_images,
        t_target,
        args.pool_size,
        layer.stride,
        layer.padding,
        fractions,
        args.device,
        args.chunk_size,
    )
    logger.info("Test cache: %.1fs, shape %s", time.time() - t0, test_cache.shape)

    # Save
    cache_path = (
        f"{model_dir}/feature_cache_step{args.step_size}_drift{args.max_drift}.pt"
    )
    import pickle

    torch.save(
        {
            "train_cache": train_cache,  # (F, num_fracs, N_train, pool_dim)
            "test_cache": test_cache,  # (F, num_fracs, N_test, pool_dim)
            "y_train": y_train,
            "y_test": y_test,
            "original_thresholds": original_thresholds.numpy(),
            "perturbation_fractions": fractions,
            "step_size": args.step_size,
            "max_drift": args.max_drift,
            "pool_size": args.pool_size,
            "t_target": t_target,
        },
        cache_path,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    size_gb = os.path.getsize(cache_path) / 1e9
    logger.info("Saved cache to %s (%.2f GB)", cache_path, size_gb)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
