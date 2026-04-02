STDP = {
    "tau_pre": 0.1,
    "tau_post": 0.1,
    "max_pre_spike_time": 1.0,
    "learning_rate": 0.1,
    "decay_factor": 1.0,
}

THRESHOLD_INIT = {
    "avg_threshold": 5.0,
    "min_threshold": 1.0,
    "std_dev": 1.0,
}

THRESHOLD_ADAPTATION = {
    "min_threshold": 1.0,
    "learning_rate": 1.0,
    "decay_factor": 1.0,
}

# Dataset-specific hyperparameter presets
_DATASET_HYPERPARAMS = {
    # MNIST-family: grayscale, 28x28, DoG encoding → 2 input channels
    "mnist": {
        "avg_threshold": 5.0,
        "min_threshold": 1.0,
        "std_dev": 1.0,
        "target_timestamp": 0.75,
        "threshold_lr": 1.0,
        "stdp_lr": 0.1,
        "annealing": 0.95,
        "in_channels": 2,
        "image_size": 28,
    },
    "fashion_mnist": {
        "avg_threshold": 5.0,
        "min_threshold": 1.0,
        "std_dev": 1.0,
        "target_timestamp": 0.75,
        "threshold_lr": 1.0,
        "stdp_lr": 0.1,
        "annealing": 0.95,
        "in_channels": 2,
        "image_size": 28,
    },
    # CIFAR-10 whitened: RGB whitened, 32x32, 6 input channels (Falez 2020 Table I)
    "cifar10_whitened": {
        "avg_threshold": 10.0,
        "min_threshold": 2.0,
        "std_dev": 0.1,
        "target_timestamp": 0.97,
        "threshold_lr": 1.0,
        "kernel_size": 5,
        "stride": 1,
        "padding": 0,
        "pool_size": 2,
        "w_min": 0.0,
        "w_max": 1.0,
        "learning_rate": 0.1,
        "beta": 1.0,
        "annealing": 0.95,
        "whitening_patch_size": 9,
        "whitening_epsilon": 1e-2,
        "whitening_rho": 0.15,
        "in_channels": 6,
        "image_size": 32,
        "num_bins": 16,
    },
    # CIFAR-10 grayscale DoG: 2 input channels
    "cifar10": {
        "avg_threshold": 5.0,
        "min_threshold": 1.0,
        "std_dev": 1.0,
        "target_timestamp": 0.7,
        "threshold_lr": 1.0,
        "annealing": 0.95,
        "in_channels": 2,
        "image_size": 32,
    },
}


def get_common_hyperparams(dataset: str) -> dict:
    """Return dataset-specific hyperparameter defaults.

    Args:
        dataset: Dataset name (e.g. 'mnist', 'cifar10_whitened', 'fashion_mnist').

    Returns:
        Dict of hyperparameters appropriate for the dataset.

    Raises:
        ValueError: If dataset name is unknown.
    """
    if dataset not in _DATASET_HYPERPARAMS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Available: {list(_DATASET_HYPERPARAMS.keys())}"
        )
    return dict(_DATASET_HYPERPARAMS[dataset])
