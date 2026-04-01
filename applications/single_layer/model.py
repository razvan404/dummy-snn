import numpy as np

from applications import default_hyperparams
from spiking.layers import IntegrateAndFireLayer
from spiking.learning import Learner, BiologicalSTDP, WinnerTakesAll
from spiking.threshold import NormalInitialization, CompetitiveThresholdAdaptation


def create_model_and_learner(setup: dict, num_inputs: int, num_outputs: int):
    model = IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        threshold_initialization=NormalInitialization(**setup["threshold_init"]),
        refractory_period=np.inf,
    )
    learner = Learner(
        model,
        learning_mechanism=BiologicalSTDP(**setup["stdp"]),
        competition=WinnerTakesAll(),
        threshold_adaptation=CompetitiveThresholdAdaptation(
            **setup["threshold_adaptation"]
        ),
    )
    return model, learner


def get_default_setup(seed: int, num_outputs: int = 100, num_epochs: int = 100):
    """Return default training setup with given seed."""
    return {
        "dataset": "mnist_subset",
        "num_epochs": num_epochs,
        "num_outputs": num_outputs,
        "seed": seed,
        "threshold_init": {**default_hyperparams.THRESHOLD_INIT},
        "threshold_adaptation": {**default_hyperparams.THRESHOLD_ADAPTATION},
        "stdp": {**default_hyperparams.STDP},
    }
