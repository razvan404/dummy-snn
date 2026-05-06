"""Compute per-filter threshold breakpoints from cumulative potentials.

For each filter f, the spike time at position (h,w) on sample n is:
    t_f(θ) = min{ tₖ : V_f(tₖ) ≥ θ }

where V_f(tₖ) is the cumulative membrane potential after all inputs at time ≤ tₖ
have arrived. V is a non-decreasing staircase, so the spike time only changes
when θ crosses one of the staircase levels.

The set of all V_f(tₖ) values across the dataset gives the exact breakpoints:
between any two consecutive breakpoints, the features (and thus accuracy) are
identical. This module collects these breakpoints to determine whether exact
piecewise optimization is feasible.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spiking import load_model
from spiking.layers import SpikingSequential
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer

logger = logging.getLogger(__name__)


@dataclass
class BreakpointStats:
    """Per-filter breakpoint statistics."""

    filter_idx: int
    num_unique: int
    num_total: int  # samples processed
    min_val: float
    max_val: float
    median_val: float
    current_threshold: float
    num_near_threshold: int  # breakpoints within ±50% of θ
    breakpoints: np.ndarray = field(repr=False)


@torch.no_grad()
def collect_breakpoints_per_filter_chunk(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    stride: int,
    padding: int,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Collect unique cumulative potential values per filter for one chunk.

    Runs the conv accumulation loop, and at each time step records all
    cumulative potential values. Returns deduplicated values per filter
    using torch.unique on GPU.

    :param input_times: (B, C, H, W) spike times.
    :param weights_4d: (F, C, kH, kW) filter weights.
    :param stride: Conv stride.
    :param padding: Conv padding.
    :param device: Computation device.
    :returns: List of F tensors (on CPU), each containing unique potential values.
    """
    input_times = input_times.to(device)
    weights_4d = weights_4d.to(device)

    B, C, H, W = input_times.shape
    F_dim = weights_4d.shape[0]
    kH = weights_4d.shape[2]
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kH) // stride + 1

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return [torch.zeros(0) for _ in range(F_dim)]

    unique_times = input_times[finite_mask].unique().sort()[0]

    cum_potential = torch.zeros(
        B, F_dim, oH, oW, dtype=input_times.dtype, device=device
    )

    # Collect cumulative potential snapshots per filter
    # To save memory, process per-filter: accumulate all time steps, then
    # collect unique values per filter in batches of time steps.
    # Strategy: store only the unique values from each time step's snapshot.
    per_filter_vals: list[list[torch.Tensor]] = [[] for _ in range(F_dim)]

    for t in unique_times:
        active = (input_times == t).float()
        contrib = F.conv2d(active, weights_4d, stride=stride, padding=padding)
        cum_potential = cum_potential + contrib

        # Extract per-filter unique values from this snapshot
        # cum_potential shape: (B, F, oH, oW)
        # For each filter, flatten spatial+batch dims and take unique
        for f in range(F_dim):
            vals = cum_potential[:, f, :, :].reshape(-1)
            positive = vals[vals > 0]
            if len(positive) > 0:
                per_filter_vals[f].append(positive.cpu())

    # Merge and deduplicate per filter
    result = []
    for f in range(F_dim):
        if per_filter_vals[f]:
            merged = torch.cat(per_filter_vals[f])
            result.append(merged.unique())
        else:
            result.append(torch.zeros(0))

    return result


@torch.no_grad()
def collect_breakpoints_per_filter_chunk_fast(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    stride: int,
    padding: int,
    device: torch.device | str = "cpu",
) -> list[np.ndarray]:
    """Fast version: collect only the FINAL cumulative potential per position.

    Key insight: the full staircase of cumulative potentials can be recovered
    from the final potential + knowledge of when each input arrived. But for
    breakpoint counting, we only need the unique values at the LAST time step
    per (sample, filter, position) — this captures the maximum potential each
    neuron can reach. The intermediate steps are a subset of these.

    Actually, intermediate steps DO matter (they're the breakpoints where
    spike time changes). So this fast version instead collects unique values
    at each time step but does dedup on GPU before transferring to CPU.

    :param input_times: (B, C, H, W) spike times.
    :param weights_4d: (F, C, kH, kW) filter weights.
    :param stride: Conv stride.
    :param padding: Conv padding.
    :param device: Computation device.
    :returns: List of F numpy arrays, each with unique potential values.
    """
    input_times = input_times.to(device)
    weights_4d = weights_4d.to(device)

    B, C, H, W = input_times.shape
    F_dim = weights_4d.shape[0]
    kH = weights_4d.shape[2]
    oH = (H + 2 * padding - kH) // stride + 1
    oW = (W + 2 * padding - kH) // stride + 1

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return [np.zeros(0, dtype=np.float32) for _ in range(F_dim)]

    unique_times = input_times[finite_mask].unique().sort()[0]

    cum_potential = torch.zeros(
        B, F_dim, oH, oW, dtype=input_times.dtype, device=device
    )

    # Preallocate buffer: (num_times, F, B*oH*oW)
    # This stores cum potential at every timestep, per filter, flattened over
    # spatial+batch. We'll unique per filter at the end.
    # Memory: num_times * F * spatial * 4 bytes
    # For cifar10: ~64 * 256 * (128*28*28) * 4 = ~50 GB — too much!
    # Instead, do incremental unique per filter across time steps on GPU.

    per_filter_unique: list[torch.Tensor] = [
        torch.zeros(0, device=device) for _ in range(F_dim)
    ]

    for t in unique_times:
        active = (input_times == t).float()
        contrib = F.conv2d(active, weights_4d, stride=stride, padding=padding)
        cum_potential = cum_potential + contrib

        # For each filter, get unique values at this time step and merge
        for f in range(F_dim):
            snapshot = cum_potential[:, f, :, :].reshape(-1)
            positive = snapshot[snapshot > 0]
            if len(positive) == 0:
                continue
            step_unique = positive.unique()
            if len(per_filter_unique[f]) > 0:
                merged = torch.cat([per_filter_unique[f], step_unique])
                per_filter_unique[f] = merged.unique()
            else:
                per_filter_unique[f] = step_unique

    return [v.cpu().numpy() for v in per_filter_unique]


def compute_breakpoints(
    model_path: str,
    loader: DataLoader,
    layer_idx: int = 0,
    device: str = "cpu",
    chunk_size: int = 256,
    max_samples: int | None = None,
) -> list[BreakpointStats]:
    """Compute per-filter breakpoint statistics from a dataset.

    :param model_path: Path to saved model.
    :param loader: DataLoader yielding (spike_times, labels).
    :param layer_idx: Which layer to analyze (for sequential models).
    :param device: Computation device.
    :param chunk_size: Batch size for processing.
    :param max_samples: Cap on number of samples to process (None = all).
    :returns: List of BreakpointStats, one per filter.
    """
    model = load_model(model_path)
    if isinstance(model, SpikingSequential):
        layer = model.layers[layer_idx]
    else:
        layer = model
    assert isinstance(layer, ConvIntegrateAndFireLayer)

    weights_4d = layer.weights_4d.detach().cpu()
    thresholds = layer.thresholds.detach().cpu().numpy()
    num_filters = layer.num_filters
    stride = layer.stride
    padding = layer.padding

    # Collect all input times
    batched = DataLoader(loader.dataset, batch_size=chunk_size, shuffle=False)
    all_times_parts = []
    total = 0
    for batch_times, _ in batched:
        all_times_parts.append(batch_times)
        total += batch_times.shape[0]
        if max_samples and total >= max_samples:
            break
    all_times = torch.cat(all_times_parts, dim=0)
    if max_samples:
        all_times = all_times[:max_samples]
    N = all_times.shape[0]
    logger.info("Processing %d samples, %d filters", N, num_filters)

    # Collect breakpoints in chunks
    # For each filter, accumulate unique values across chunks using numpy merge
    per_filter_arrays: list[list[np.ndarray]] = [[] for _ in range(num_filters)]

    n_chunks = (N + chunk_size - 1) // chunk_size
    for start in tqdm(
        range(0, N, chunk_size), total=n_chunks, desc="Collecting breakpoints"
    ):
        end = min(start + chunk_size, N)
        chunk = all_times[start:end]

        chunk_bps = collect_breakpoints_per_filter_chunk_fast(
            chunk, weights_4d, stride, padding, device=device
        )

        for f in range(num_filters):
            if len(chunk_bps[f]) > 0:
                per_filter_arrays[f].append(chunk_bps[f])

        # Free GPU memory
        if device != "cpu":
            torch.cuda.empty_cache()

    # Final merge: numpy unique across all chunks per filter
    logger.info("Merging breakpoints across chunks...")
    results = []
    for f in tqdm(range(num_filters), desc="Merging per-filter"):
        if per_filter_arrays[f]:
            merged = np.unique(np.concatenate(per_filter_arrays[f]))
        else:
            merged = np.array([], dtype=np.float32)

        theta = float(thresholds[f])
        lower = theta * 0.5
        upper = theta * 1.5
        near_mask = (merged >= lower) & (merged <= upper)

        stats = BreakpointStats(
            filter_idx=f,
            num_unique=len(merged),
            num_total=N,
            min_val=float(merged[0]) if len(merged) > 0 else 0.0,
            max_val=float(merged[-1]) if len(merged) > 0 else 0.0,
            median_val=float(np.median(merged)) if len(merged) > 0 else 0.0,
            current_threshold=theta,
            num_near_threshold=int(near_mask.sum()),
            breakpoints=merged,
        )
        results.append(stats)

    return results


def print_breakpoint_summary(stats_list: list[BreakpointStats]) -> None:
    """Print a concise summary table of breakpoint statistics."""
    print(
        f"\n{'Filter':>6} {'Unique BPs':>10} {'Near θ (±50%)':>14} "
        f"{'θ current':>10} {'BP min':>10} {'BP median':>10} {'BP max':>10}"
    )
    print("-" * 80)

    total_unique = 0
    total_near = 0
    for s in stats_list:
        print(
            f"{s.filter_idx:>6d} {s.num_unique:>10,d} {s.num_near_threshold:>14,d} "
            f"{s.current_threshold:>10.2f} {s.min_val:>10.2f} "
            f"{s.median_val:>10.2f} {s.max_val:>10.2f}"
        )
        total_unique += s.num_unique
        total_near += s.num_near_threshold

    print("-" * 80)
    print(f"{'TOTAL':>6} {total_unique:>10,d} {total_near:>14,d}")
    print(f"\nSamples processed: {stats_list[0].num_total:,d}")
    print(f"Filters: {len(stats_list)}")
    print(f"Avg unique breakpoints per filter: {total_unique / len(stats_list):,.0f}")
    print(f"Avg breakpoints near θ per filter: {total_near / len(stats_list):,.0f}")


def save_breakpoint_results(
    stats_list: list[BreakpointStats], output_path: str
) -> None:
    """Save breakpoint analysis results to JSON (without raw breakpoint arrays)."""
    records = []
    for s in stats_list:
        if len(s.breakpoints) > 0:
            ratios = s.breakpoints / s.current_threshold
            hist_counts, hist_edges = np.histogram(
                ratios, bins=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf]
            )
            ratio_histogram = {
                f"{hist_edges[i]:.2f}-{hist_edges[i + 1]:.2f}": int(hist_counts[i])
                for i in range(len(hist_counts))
            }
        else:
            ratio_histogram = {}

        records.append(
            {
                "filter_idx": s.filter_idx,
                "num_unique_breakpoints": s.num_unique,
                "num_near_threshold": s.num_near_threshold,
                "current_threshold": s.current_threshold,
                "min_breakpoint": s.min_val,
                "max_breakpoint": s.max_val,
                "median_breakpoint": s.median_val,
                "ratio_histogram": ratio_histogram,
            }
        )

    output = {
        "num_samples": stats_list[0].num_total,
        "num_filters": len(stats_list),
        "total_unique_breakpoints": sum(s.num_unique for s in stats_list),
        "filters": records,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved breakpoint analysis to %s", output_path)
