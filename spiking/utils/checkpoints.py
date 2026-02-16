import os

import torch

from spiking.spiking_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model, path)


def load_model(path: str):
    return torch.load(path, weights_only=False)
