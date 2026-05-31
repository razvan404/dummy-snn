import os

import torch

from spikinn.spikinn_module import SpikinnModule


def save_model(model: SpikinnModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


def _fix_buffer_grad(module: SpikinnModule) -> None:
    # In-place ops during training can promote buffer requires_grad; restore invariant.
    for buf in module.buffers():
        buf.requires_grad_(False)


def load_model(path: str) -> SpikinnModule:
    model = torch.load(path, weights_only=False)
    _fix_buffer_grad(model)
    return model
