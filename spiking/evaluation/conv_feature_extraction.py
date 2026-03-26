import numpy as np
import torch
from torch.utils.data import DataLoader

from spiking.evaluation.feature_extraction import spike_times_to_features
from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer


def sum_pool_features(features: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    """Divide spatial feature maps into a pool_size × pool_size grid and sum each region.

    For pool_size=2 with 28×28 input: divides into 4 quadrants of 14×14 each,
    sums each quadrant → output 2×2 per filter.

    Args:
        features: (F, oH, oW) or (B, F, oH, oW) feature tensor.
        pool_size: Number of regions per spatial dimension.

    Returns:
        Tensor with spatial dims equal to pool_size × pool_size.
    """
    if pool_size == 1:
        return features

    needs_batch = features.dim() == 3
    if needs_batch:
        features = features.unsqueeze(0)

    B, F_dim, H, W = features.shape
    rH, rW = H // pool_size, W // pool_size
    # Trim to exact multiple and reshape into grid of regions
    trimmed = features[:, :, : rH * pool_size, : rW * pool_size]
    pooled = trimmed.reshape(B, F_dim, pool_size, rH, pool_size, rW).sum(dim=(3, 5))

    if needs_batch:
        pooled = pooled.squeeze(0)
    return pooled


@torch.no_grad()
def extract_conv_features(
    model: ConvIntegrateAndFireLayer,
    dataloader: DataLoader,
    pool_size: int = 2,
    t_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run conv model inference and extract flat feature vectors.

    Uses batched analytical spike time computation, then applies
    spike_times_to_features and sum pooling.

    Args:
        model: Convolutional spiking layer.
        dataloader: DataLoader yielding (times, labels) per sample.
        pool_size: Sum pooling window size.
        t_target: Optional target time for feature conversion.

    Returns:
        (X, y) where X is (N, flat_features) and y is (N,) numpy arrays.
    """
    model.eval()

    chunk_loader = DataLoader(dataloader.dataset, batch_size=256, shuffle=False)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for batch_times, batch_labels in chunk_loader:
        spike_times = model.infer_spike_times_batch(batch_times)
        features = spike_times_to_features(spike_times, t_target)
        pooled = sum_pool_features(features, pool_size)
        X_parts.append(pooled.flatten(1).numpy())
        y_parts.append(batch_labels.numpy())

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)
