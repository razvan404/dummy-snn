import numpy as np
import torch
from torch.utils.data import DataLoader

from spiking import SpikingModule
from spiking.evaluation.decoding import ScaledInversion, TargetRelative


def spike_times_to_features(
    spike_times: torch.Tensor,
    t_target: float | None = None,
) -> torch.Tensor:
    """Convert spike times to feature values in [0, 1].

    Without t_target: scaled inversion, clamp((1 - t) / (1 - min_t), 0, 1).
    With t_target: Falez Eq 10, clamp(1 - (t - t_target) / (1 - t_target), 0, 1).
    """
    decoder = TargetRelative(t_target) if t_target is not None else ScaledInversion()
    return decoder.decode(spike_times)


@torch.no_grad()
def extract_spike_times(
    model: SpikingModule,
    dataloader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model inference on a dataloader and return raw spike times and labels.

    Returns (spike_times, labels) as torch tensors.
    """
    model.eval()
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    all_times, all_labels = next(iter(full_loader))
    spike_times = model.infer_spike_times_batch(all_times.flatten(1))
    return spike_times, all_labels


@torch.no_grad()
def extract_features(
    model: SpikingModule,
    dataloader: DataLoader,
    t_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a dataloader and return (X, y) numpy arrays.

    Uses batched analytical spike time computation for efficiency.
    Processes the full dataset in one batch for optimal matmul throughput.
    """
    model.eval()
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    all_times, all_labels = next(iter(full_loader))
    spike_times = model.infer_spike_times_batch(all_times.flatten(1))
    X = spike_times_to_features(spike_times, t_target).numpy()
    y = all_labels.numpy()
    return X, y
