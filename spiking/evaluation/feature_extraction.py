import numpy as np
import torch
from torch.utils.data import DataLoader

from spiking import SpikingModule, iterate_spikes


@torch.no_grad()
def extract_features(
    model: SpikingModule,
    dataloader: DataLoader,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a dataloader and return (X, y) numpy arrays.

    Features are computed as clamp(1.0 - spike_times, 0, 1).
    Breaks early per sample when all neurons have already spiked.
    """
    X, y = [], []
    model.eval()
    for spikes, label, _ in dataloader:
        for incoming_spikes, current_time, dt in iterate_spikes(spikes, shape=shape):
            model.forward(incoming_spikes.flatten(), current_time=current_time, dt=dt)
            if torch.all(torch.isfinite(model.spike_times)):
                break
        X.append(torch.clamp(1.0 - model.spike_times, min=0, max=1.0).numpy())
        y.append(label)
        model.reset()
    return np.array(X), np.array(y)
