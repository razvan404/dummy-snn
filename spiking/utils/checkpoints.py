import pickle

from spiking.spiking_module import SpikingModule


def save_model(model: SpikingModule, path: str):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path: str):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model
