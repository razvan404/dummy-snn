import os

import torch

from spiking.spiking_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


def _fix_buffer_grad(module: SpikingModule, requires_grad: bool):
    """In-place ops during training can promote buffer requires_grad to True.
    After deserialization, restore the invariant: buffers don't require grad.
    SpikingSequential stores layers in a plain list, so nn.Module.buffers()
    doesn't traverse them — we walk the layers attribute manually.
    """
    for buf in module.buffers():
        buf.requires_grad_(requires_grad)
    if hasattr(module, "layers"):
        for layer in module.layers:
            _fix_buffer_grad(layer, requires_grad)


def load_model(path: str, requires_grad: bool = False):
    model = torch.load(path, weights_only=False)
    _fix_buffer_grad(model, requires_grad)
    return model
