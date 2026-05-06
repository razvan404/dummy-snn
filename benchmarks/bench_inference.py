"""Wall-clock comparison of dense vs sparse-event spiking inference.

Times both paths on a realistic CIFAR-10 ZCA forward pass and a Fashion-MNIST
DoG forward pass at increasing batch sizes. Reports total time and per-image
throughput; verifies spike-time agreement at each batch size.

Run: ``python -m benchmarks.bench_inference``
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch

import spiking_backend
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.threshold.normal_initialization import NormalInitialization


@dataclass
class Profile:
    name: str
    in_channels: int
    H: int
    W: int
    num_filters: int
    kernel_size: int
    stride: int
    padding: int
    num_bins: int
    sparsity: float
    avg_threshold: float


PROFILES = {
    "cifar10": Profile(
        name="cifar10_whitened",
        in_channels=6,
        H=32,
        W=32,
        num_filters=256,
        kernel_size=5,
        stride=1,
        padding=0,
        num_bins=64,
        sparsity=0.5,
        avg_threshold=10.0,
    ),
    "fmnist": Profile(
        name="fashion_mnist_dog",
        in_channels=2,
        H=28,
        W=28,
        num_filters=256,
        kernel_size=5,
        stride=1,
        padding=0,
        num_bins=64,
        sparsity=0.5,
        avg_threshold=5.0,
    ),
}


def _build_layer(p: Profile) -> ConvIntegrateAndFireLayer:
    init = NormalInitialization(
        avg_threshold=p.avg_threshold, std_dev=0.5, min_threshold=0.5
    )
    layer = ConvIntegrateAndFireLayer(
        in_channels=p.in_channels,
        num_filters=p.num_filters,
        kernel_size=p.kernel_size,
        stride=p.stride,
        padding=p.padding,
        threshold_initialization=init,
        refractory_period=float("inf"),
    )
    torch.nn.init.uniform_(layer.weights, a=0.0, b=1.0)
    return layer


def _make_batch(p: Profile, batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    times = (
        torch.randint(0, p.num_bins, (batch, p.in_channels, p.H, p.W), generator=g).float()
        / p.num_bins
    )
    mask = torch.rand(times.shape, generator=g) < p.sparsity
    return torch.where(mask, times, torch.full_like(times, float("inf")))


def _time(fn: Callable[[], object], runs: int, sync_cuda: bool = False) -> float:
    """Best-of-N wall-clock seconds (excludes warm-up).

    If ``sync_cuda`` is True, ``torch.cuda.synchronize()`` brackets each call
    so the timing reflects actual GPU work rather than kernel-launch return.
    """
    fn()
    if sync_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return min(samples)


def _max_bin_diff(a: torch.Tensor, b: torch.Tensor, num_bins: int) -> float:
    """Return the spike-time disagreement, measured in discretised bins."""
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not finite.any():
        return 0.0
    return (a[finite] - b[finite]).abs().max().item() * num_bins


def bench(profile_key: str, batch_sizes: list[int], runs: int, device: str) -> None:
    p = PROFILES[profile_key]
    print(
        f"\n=== {p.name} on {device} (C={p.in_channels} H={p.H} W={p.W} "
        f"F={p.num_filters} k={p.kernel_size} bins={p.num_bins} "
        f"sparsity={p.sparsity}) ==="
    )
    layer = _build_layer(p).to(device)
    sync = device == "cuda"

    print(
        f"  {'batch':>6}  {'dense ms':>10}  {'sparse ms':>10}  {'speedup':>8}  "
        f"{'dense img/s':>12}  {'sparse img/s':>13}  {'max bins off':>13}"
    )
    for B in batch_sizes:
        times = _make_batch(p, B, seed=B).to(device)

        def dense() -> tuple[torch.Tensor, torch.Tensor]:
            return layer._conv2d_accumulate(times)

        def sparse() -> tuple[torch.Tensor, torch.Tensor]:
            return spiking_backend.spike_driven_conv_accumulate(
                times,
                layer.weights_4d,
                layer.thresholds,
                stride=p.stride,
                padding=p.padding,
            )

        t_dense = _time(dense, runs, sync_cuda=sync)
        t_sparse = _time(sparse, runs, sync_cuda=sync)
        speedup = t_dense / t_sparse if t_sparse > 0 else float("inf")
        a_st, _ = dense()
        b_st, _ = sparse()
        max_diff = _max_bin_diff(a_st, b_st, p.num_bins)

        print(
            f"  {B:>6}  {t_dense*1000:>10.1f}  {t_sparse*1000:>10.1f}  "
            f"{speedup:>7.2f}x  {B/t_dense:>12.1f}  {B/t_sparse:>13.1f}  "
            f"{max_diff:>13.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES.keys()),
        choices=list(PROFILES.keys()),
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    print(
        f"spiking_backend compiled extension: "
        f"{'AVAILABLE' if spiking_backend.is_compiled_available() else 'unavailable, falling back to reference'}"
    )
    print(
        f"PyTorch threads: {torch.get_num_threads()}  CUDA: {torch.cuda.is_available()}"
    )
    for dev in args.devices:
        for prof in args.profiles:
            bench(prof, args.batches, args.runs, device=dev)


if __name__ == "__main__":
    main()
