import numpy as np
import torch
from torch.utils.data import DataLoader

from spikinn import SpikingModule
from spikinn.evaluation.decoding import Decoder, ScaledInversion, TargetRelative


def spike_times_to_features(
    spike_times: torch.Tensor,
    t_target: float | None = None,
    decoder: Decoder | None = None,
) -> torch.Tensor:
    """Convert spike times to [0, 1] features.

    Pass an explicit ``decoder`` to override the default; otherwise falls back to
    :class:`TargetRelative` when ``t_target`` is given, else :class:`ScaledInversion`.
    """
    if decoder is None:
        decoder = TargetRelative(t_target) if t_target is not None else ScaledInversion()
    return decoder.decode(spike_times)


@torch.no_grad()
def extract_features(
    model: SpikingModule,
    dataloader: DataLoader,
    t_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on full dataset in one batch."""
    model.eval()
    full_loader = DataLoader(
        dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False
    )
    all_times, all_labels = next(iter(full_loader))
    spike_times = model.infer_spike_times_batch(all_times.flatten(1))
    X = spike_times_to_features(spike_times, t_target).numpy()
    y = all_labels.numpy()
    return X, y
