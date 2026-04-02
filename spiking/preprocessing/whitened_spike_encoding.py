import torch

from spiking.preprocessing.latency_encoding import apply_latency_encoding


def encode_whitened_image(whitened_image: torch.Tensor) -> torch.Tensor:
    """Encode a whitened image into spike times (Falez 2020 Section IV-A).

    1. Scale per-sample to [-1, 1] via min-max normalization.
    2. Split into positive (X+) and negative (X-) channels, interleaved.
    3. Apply latency encoding: t = 1 - x (brighter spikes earlier).

    :param whitened_image: (C, H, W) whitened image tensor.
    :returns: (2*C, H, W) spike times. Channel order: [C0+, C0-, C1+, C1-, ...].
    """
    C, H, W = whitened_image.shape

    # Per-sample min-max scaling to [-1, 1] (paper Section IV-A step 2)
    mn = whitened_image.min()
    mx = whitened_image.max()
    denom = mx - mn
    if denom > 0:
        scaled = 2.0 * (whitened_image - mn) / denom - 1.0
    else:
        scaled = whitened_image

    # Split into positive and negative channels, interleaved
    pos = torch.clamp(scaled, min=0.0)  # (C, H, W)
    neg = torch.clamp(-scaled, min=0.0)  # (C, H, W)
    interleaved = torch.stack([pos, neg], dim=1)  # (C, 2, H, W)
    split = interleaved.reshape(2 * C, H, W)

    # Apply latency encoding: t = 1 - x, zero intensity → inf
    return apply_latency_encoding(split)
