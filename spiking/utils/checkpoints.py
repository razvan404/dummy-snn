import os

import torch

from spiking.spiking_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


def _fix_buffer_grad(module: SpikingModule) -> None:
    """In-place ops during training can promote buffer requires_grad to True.
    After deserialization, restore the invariant: buffers don't require grad.
    """
    for buf in module.buffers():
        buf.requires_grad_(False)


def load_model(path: str) -> SpikingModule:
    model = torch.load(path, weights_only=False)
    _fix_buffer_grad(model)
    return model
