import json
import math
import os

import numpy as np
from torch.utils.data import DataLoader

from applications.deep_linear.model import create_model
from applications import default_hyperparams
from spiking import (
    Learner,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    TargetTimestampAdaptation,
    SequentialThresholdAdaptation,
)
from spiking.layers import SpikingSequential

T_OBJECTIVES = [0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.95]
SEEDS = [1, 2, 3, 4, 5]
NUM_EPOCHS = 3
NUM_OUTPUTS = 256
LR_DECAY = 0.95  # Per-epoch learning rate decay for STDP and threshold adaptation


def create_model_and_learner(
    num_inputs: int,
    t_objective: float,
    epsilon: float = 0.0,
) -> tuple[SpikingSequential, object, Learner]:
    """Create single-layer IF model with competitive + target-timestamp adaptation.

    Returns (sub_model, layer, learner).
    """
    setup = {
        "threshold_init": {
            "avg_threshold": num_inputs / 20,
            "min_threshold": 1.0,
            "std_dev": 1.0,
        },
    }
    model = create_model(setup, num_inputs, [NUM_OUTPUTS])
    layer = model.layers[0]

    threshold_params = {
        **default_hyperparams.THRESHOLD_ADAPTATION,
        "decay_factor": LR_DECAY,
    }
    adaptation = SequentialThresholdAdaptation(
        [
            CompetitiveThresholdAdaptation(**threshold_params),
            TargetTimestampAdaptation(
                target_timestamp=t_objective,
                epsilon=epsilon,
                **threshold_params,
            ),
        ]
    )

    stdp_params = {**default_hyperparams.STDP, "decay_factor": LR_DECAY}
    learner = Learner(
        layer,
        learning_mechanism=STDP(**stdp_params),
        competition=WinnerTakesAll(),
        threshold_adaptation=adaptation,
    )

    sub_model = SpikingSequential(*model.layers[:1])
    return sub_model, layer, learner


def output_dir_for(base_dir: str, t_obj: float, seed: int) -> str:
    return f"{base_dir}/tobj_{t_obj}/seed_{seed}"


def find_trained_models(base_dir: str) -> list[tuple[str, float, int]]:
    """Discover all trained models under base_dir.

    Returns list of (model_path, t_obj, seed) sorted by (t_obj, seed).
    """
    models = []
    if not os.path.isdir(base_dir):
        return models
    for tobj_name in sorted(os.listdir(base_dir)):
        if not tobj_name.startswith("tobj_"):
            continue
        try:
            t_obj = float(tobj_name.split("_", 1)[1])
        except ValueError:
            continue
        tobj_path = os.path.join(base_dir, tobj_name)
        for seed_name in sorted(os.listdir(tobj_path)):
            if not seed_name.startswith("seed_"):
                continue
            try:
                seed = int(seed_name.split("_", 1)[1])
            except ValueError:
                continue
            model_path = os.path.join(tobj_path, seed_name, "model.pth")
            if os.path.exists(model_path):
                models.append((model_path, t_obj, seed))
    return models


def save_feature_group(path: str, features: dict[str, np.ndarray]) -> None:
    data = {k: v.tolist() for k, v in features.items()}
    with open(path, "w") as f:
        json.dump(data, f)


def load_feature_group(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}
