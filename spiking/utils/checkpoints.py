import os
import pickle

from spiking.spiking_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model
