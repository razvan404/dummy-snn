from applications.default_hyperparams import get_common_hyperparams

# Shared perturbation fractions: -0.5 to +0.25 in 25 steps, excluding 0.0
PERTURBATION_FRACTIONS = [
    round(-0.5 + i * (0.75 / 24), 4)
    for i in range(25)
    if round(-0.5 + i * (0.75 / 24), 4) != 0.0
]


# Dataset-specific perturbation parameters
_PERTURBATION_PARAMS = {
    "mnist": {
        "is_conv": False,
        "pool_size": 1,
    },
    "fashion_mnist": {
        "is_conv": False,
        "pool_size": 1,
    },
    "cifar10": {
        "is_conv": False,
        "pool_size": 1,
    },
    "cifar10_whitened": {
        "is_conv": True,
        "pool_size": 2,
    },
}


def get_perturbation_params(dataset: str) -> dict:
    """Return perturbation sweep parameters for a dataset.

    Merges dataset-specific perturbation params with common hyperparams.

    Returns dict with keys: is_conv, pool_size, target_timestamp, in_channels,
    image_size, and any other common hyperparams.
    """
    common = get_common_hyperparams(dataset)
    overrides = _PERTURBATION_PARAMS.get(dataset, {"is_conv": False, "pool_size": 1})
    return {**common, **overrides}
