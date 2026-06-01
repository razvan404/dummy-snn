import os

import torch

from spikinn.spikinn_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


def _fix_buffer_grad(module: SpikingModule) -> None:
    # In-place ops during training can promote buffer requires_grad; restore invariant.
    for buf in module.buffers():
        buf.requires_grad_(False)


def load_model(path: str) -> SpikingModule:
    import sys
    import spikinn
    sys.modules["spiking"] = spikinn
    for sub in ["layers", "learning", "threshold", "spikinn_module", "utils", "training"]:
        try:
            mod = __import__(f"spikinn.{sub}", fromlist=["*"])
            sys.modules[f"spiking.{sub}"] = mod
        except ImportError:
            pass

    model = torch.load(path, weights_only=False)
    _fix_buffer_grad(model)
    return model
