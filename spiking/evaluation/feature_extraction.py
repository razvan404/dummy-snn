import numpy as np
import torch
from torch.utils.data import DataLoader

from spiking import SpikingModule


def spike_times_to_features(
    spike_times: torch.Tensor,
    t_target: float | None = None,
) -> torch.Tensor:
    """Convert spike times to feature values in [0, 1].

    Without t_target: linear inversion, clamp(1 - t, 0, 1).
    With t_target: Falez Eq 10, clamp(1 - (t - t_target) / (1 - t_target), 0, 1).
    """
    if t_target is None:
        return torch.clamp(1.0 - spike_times, min=0, max=1.0)
    return torch.clamp(
        1.0 - (spike_times - t_target) / (1.0 - t_target), min=0, max=1.0
    )


@torch.no_grad()
def extract_features(
    model: SpikingModule,
    dataloader: DataLoader,
    shape: tuple[int, int, int],
    t_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a dataloader and return (X, y) numpy arrays.

    Uses analytical spike time computation (no iterative forward pass).
    """
    X, y = [], []
    model.eval()
    for times, label in dataloader:
        spike_times = model.infer_spike_times(times.flatten())
        X.append(spike_times_to_features(spike_times, t_target).numpy())
        y.append(label)
    return np.array(X), np.array(y)
