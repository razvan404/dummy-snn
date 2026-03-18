import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spiking.layers.conv_integrate_and_fire import ConvIntegrateAndFireLayer
from spiking.evaluation.feature_extraction import spike_times_to_features


def sum_pool_features(features: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    """Apply sum pooling to spatial feature maps.

    Args:
        features: (F, oH, oW) or (B, F, oH, oW) feature tensor.
        pool_size: Spatial pooling window size.

    Returns:
        Pooled tensor with spatial dims divided by pool_size.
    """
    if pool_size == 1:
        return features

    needs_batch = features.dim() == 3
    if needs_batch:
        features = features.unsqueeze(0)

    # Sum pooling = avg_pool * pool_size^2
    pooled = F.avg_pool2d(features.float(), pool_size) * (pool_size**2)

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

    # Collect all data into a single batch
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    all_times, all_labels = next(iter(full_loader))

    # Batched analytical inference: (B, C, H, W) → (B, F, oH, oW)
    spike_times = model.infer_spike_times_batch(all_times)

    # Convert spike times to features
    features = spike_times_to_features(spike_times, t_target)  # (B, F, oH, oW)

    # Apply sum pooling
    pooled = sum_pool_features(features, pool_size)  # (B, F, pH, pW)

    # Flatten spatial dims
    X = pooled.flatten(1).numpy()  # (B, F * pH * pW)
    y = all_labels.numpy()
    return X, y
