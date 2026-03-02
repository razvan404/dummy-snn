import numpy as np
import torch
from torch.utils.data import DataLoader

from spiking import SpikingModule


@torch.no_grad()
def extract_features(
    model: SpikingModule,
    dataloader: DataLoader,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a dataloader and return (X, y) numpy arrays.

    Features are computed as clamp(1.0 - spike_times, 0, 1).
    Uses analytical spike time computation (no iterative forward pass).
    """
    X, y = [], []
    model.eval()
    for times, label in dataloader:
        spike_times = model.infer_spike_times(times.flatten())
        X.append(torch.clamp(1.0 - spike_times, min=0, max=1.0).numpy())
        y.append(label)
    return np.array(X), np.array(y)
