import torch


def sum_pool_features(features: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
    """Divide spatial feature maps into a pool_size × pool_size grid and sum each region.

    For pool_size=2 with 28×28 input: divides into 4 quadrants of 14×14 each,
    sums each quadrant → output 2×2 per filter.

    :param features: (F, oH, oW) or (B, F, oH, oW) feature tensor.
    :param pool_size: Number of regions per spatial dimension.
    :returns: Tensor with spatial dims equal to pool_size × pool_size.
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
