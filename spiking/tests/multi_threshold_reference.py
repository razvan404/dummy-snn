import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spiking.evaluation.conv_feature_extraction import sum_pool_features
from spiking.evaluation.feature_extraction import spike_times_to_features


def collect_conv_input_times(
    loader: DataLoader,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ``(times, labels)`` from a loader; preserves spatial dims."""
    batched = DataLoader(loader.dataset, batch_size=chunk_size, shuffle=False)
    time_parts = []
    label_parts = []
    for batch_times, batch_labels in batched:
        time_parts.append(batch_times)
        label_parts.append(batch_labels)
    return torch.cat(time_parts, dim=0), torch.cat(label_parts, dim=0)


@torch.no_grad()
def multi_threshold_conv_accumulate(
    input_times: torch.Tensor,
    weights_4d: torch.Tensor,
    thresholds_2d: torch.Tensor,
    stride: int,
    padding: int,
    device: torch.device | str = "cpu",
    backend: str = "dense",
    num_bins: int = 64,
) -> torch.Tensor:
    """Single-pass conv accumulation across multiple threshold sets."""
    input_times = input_times.to(device)
    weights_4d = weights_4d.to(device)
    thresholds_2d = thresholds_2d.to(device)

    if backend == "gather":
        from spiking_backend import first_spike_times_multi_threshold

        return first_spike_times_multi_threshold(
            input_times,
            weights_4d,
            thresholds_2d,
            num_bins=num_bins,
            stride=stride,
            padding=padding,
        ).cpu()
    if backend != "dense":
        raise ValueError(
            f"backend must be 'dense' or 'gather'; got {backend!r}"
        )

    B, C, H, W = input_times.shape
    num_fracs, num_filters = thresholds_2d.shape
    kernel_size = weights_4d.shape[2]
    oH = (H + 2 * padding - kernel_size) // stride + 1
    oW = (W + 2 * padding - kernel_size) // stride + 1

    result = torch.full(
        (num_fracs, B, num_filters, oH, oW),
        float("inf"),
        dtype=input_times.dtype,
        device=device,
    )
    not_yet_spiked = torch.ones(
        (num_fracs, B, num_filters, oH, oW),
        dtype=torch.bool,
        device=device,
    )
    cum_potential = torch.zeros(
        (B, num_filters, oH, oW),
        dtype=input_times.dtype,
        device=device,
    )
    thresholds_5d = thresholds_2d.view(num_fracs, 1, num_filters, 1, 1)

    finite_mask = torch.isfinite(input_times)
    if not finite_mask.any():
        return result.cpu()

    unique_times = input_times[finite_mask].unique().sort()[0]

    for t in unique_times:
        active = (input_times == t).float()
        contrib = F.conv2d(active, weights_4d, stride=stride, padding=padding)
        cum_potential += contrib

        crossed = (cum_potential.unsqueeze(0) >= thresholds_5d) & not_yet_spiked
        result[crossed] = t
        not_yet_spiked &= ~crossed

        if not not_yet_spiked.any():
            break

    return result.cpu()


def _spike_times_to_pooled_features(
    spike_times: torch.Tensor,
    t_target: float | None,
    pool_size: int,
) -> np.ndarray:
    num_fracs, B, F_dim, oH, oW = spike_times.shape
    flat = spike_times.reshape(num_fracs * B, F_dim, oH, oW)
    features = spike_times_to_features(flat, t_target)
    pooled = sum_pool_features(features, pool_size)
    flat_features = pooled.flatten(1).numpy()
    return flat_features.reshape(num_fracs, B, -1)
