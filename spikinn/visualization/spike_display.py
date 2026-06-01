import numpy as np
import torch


def dog_to_rgb(encoded: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert Difference-of-Gaussians encoded spike times to an RGB visualization.

    ON spikes (channel 0) are mapped to Green; OFF spikes (channel 1) to Red.
    Earlier spike times are represented as brighter pixels.
    """
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().numpy()

    # Input shape: (2, H, W)
    H, W = encoded.shape[1], encoded.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # intensity = max(0, 1 - time)
    # ON channel (index 0) -> Green
    rgb[..., 1] = np.clip(1.0 - encoded[0], 0.0, 1.0)
    # OFF channel (index 1) -> Red
    rgb[..., 0] = np.clip(1.0 - encoded[1], 0.0, 1.0)
    return rgb


def whitened_to_rgb(encoded: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert ZCA-whitened interleaved ON/OFF spike times to an RGB visualization.

    Positive/ON channels (R+, G+, B+ at indices 0, 2, 4) are mapped to RGB.
    Earlier spike times are represented as brighter pixels.
    """
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().numpy()

    # Input shape: (6, H, W) or (2*C, H, W)
    C = encoded.shape[0] // 2
    H, W = encoded.shape[1], encoded.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # R+ (0) -> Red, G+ (2) -> Green, B+ (4) -> Blue
    if C >= 3:
        rgb[..., 0] = np.clip(1.0 - encoded[0], 0.0, 1.0)
        rgb[..., 1] = np.clip(1.0 - encoded[2], 0.0, 1.0)
        rgb[..., 2] = np.clip(1.0 - encoded[4], 0.0, 1.0)
    else:
        # Fallback to standard DoG mapping if C < 3
        rgb[..., 1] = np.clip(1.0 - encoded[0], 0.0, 1.0)
        if encoded.shape[0] > 1:
            rgb[..., 0] = np.clip(1.0 - encoded[1], 0.0, 1.0)
    return rgb
