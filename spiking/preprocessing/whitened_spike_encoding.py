import torch

from spiking.preprocessing.latency_encoding import apply_latency_encoding


def encode_whitened_image(whitened_image: torch.Tensor) -> torch.Tensor:
    """Encode a whitened image into spike times (Falez 2020 Section IV-A).

    1. Scale per-sample to [-1, 1].
    2. Split into positive (X+) and negative (X-) channels.
    3. Apply latency encoding: t = 1 - x (brighter spikes earlier).

    Args:
        whitened_image: (C, H, W) whitened image tensor.

    Returns:
        (2*C, H, W) spike times. Channel order: [C0+, C1+, ..., C0-, C1-, ...].
    """
    C, H, W = whitened_image.shape

    # Per-sample scaling to [-1, 1]
    abs_max = whitened_image.abs().max()
    if abs_max > 0:
        scaled = whitened_image / abs_max
    else:
        scaled = whitened_image

    # Split into positive and negative channels
    pos = torch.clamp(scaled, min=0.0)  # (C, H, W)
    neg = torch.clamp(-scaled, min=0.0)  # (C, H, W)

    # Stack: [C0+, C1+, ..., C0-, C1-, ...]
    split = torch.cat([pos, neg], dim=0)  # (2*C, H, W)

    # Apply latency encoding: t = 1 - x, zero intensity → inf
    return apply_latency_encoding(split)
