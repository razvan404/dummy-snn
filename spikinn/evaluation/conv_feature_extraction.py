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


def pooled_features(
    spike_times: torch.Tensor,
    t_target: float | None = None,
    pool_size: int = 2,
    decoder=None,
) -> torch.Tensor:
    """Canonical pooled-feature step (decode spike times then spatial sum-pool).

    Single definition shared by evaluate and the feature-cache builder so the
    decode→pool pipeline cannot drift between them. Pass an explicit ``decoder``
    (e.g. :class:`~spikinn.evaluation.decoding.TargetRelativeSigmoid`) to override the
    default hard ``TargetRelative`` decode.
    """
    from spikinn.evaluation.feature_extraction import spike_times_to_features

    return sum_pool_features(
        spike_times_to_features(spike_times, t_target=t_target, decoder=decoder), pool_size
    )
