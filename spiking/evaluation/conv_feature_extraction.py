import torch


def sum_pool_features(features: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    """Spatial sum-pooling into pool_size×pool_size grid."""
    if pool_size == 1:
        return features

    needs_batch = features.dim() == 3
    if needs_batch:
        features = features.unsqueeze(0)

    B, F_dim, H, W = features.shape
    rH, rW = H // pool_size, W // pool_size
    trimmed = features[:, :, : rH * pool_size, : rW * pool_size]
    pooled = trimmed.reshape(B, F_dim, pool_size, rH, pool_size, rW).sum(dim=(3, 5))

    if needs_batch:
        pooled = pooled.squeeze(0)
    return pooled
