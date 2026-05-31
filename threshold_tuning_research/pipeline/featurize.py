from __future__ import annotations

import torch

from spikinn.spikinn_module import SpikinnModule


@torch.no_grad()
def featurize_through(
    model: SpikinnModule,
    images: torch.Tensor,
    *,
    device: str = "cpu",
    chunk_size: int = 256,
) -> torch.Tensor:
    """Run a frozen model over images in chunks; return its output spike-time maps on CPU.

    Used to produce the frozen prefix's (layer-1 + min-pool) output that layer-2 trains
    on and that the layer-2 feature cache is built from.
    """
    model = model.to(device).eval()
    outs = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size].to(device)
        outs.append(model.infer_spike_times_batch(chunk).cpu())
    return torch.cat(outs, dim=0)
