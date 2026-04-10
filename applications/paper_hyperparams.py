from dataclasses import dataclass


@dataclass(frozen=True)
class MnistHyperparams:
    """Falez 2019 (Paper 19) Table I — MNIST with DoG preprocessing."""

    # Preprocessing
    sigma_center: float = 1.0
    sigma_surround: float = 4.0
    in_channels: int = 2  # ON/OFF from DoG

    # Architecture
    kernel_size: int = 5
    stride: int = 1
    padding: int = 0
    num_filters: int = 32
    pool_size: int = 2

    # STDP
    stdp_variant: str = "biological"
    stdp_lr: float = 0.1
    biological_tau: float = 0.1
    w_min: float = 0.0
    w_max: float = 1.0

    # Threshold
    threshold_avg: float = 5.0
    threshold_std: float = 1.0
    min_threshold: float = 1.0
    threshold_lr: float = 1.0
    target_timestamp: float = 0.75

    # Training
    annealing: float = 0.95
    num_epochs: int = 100
    num_patches: int = 50
    patch_size: int = 5


@dataclass(frozen=True)
class Cifar10Hyperparams:
    """Falez 2020 (Paper 20) Table I — CIFAR-10 with ZCA whitening."""

    # Preprocessing
    whitening_patch_size: int = 9
    whitening_epsilon: float = 1e-2
    whitening_rho: float = 1.0
    in_channels: int = 6  # R+/R-/G+/G-/B+/B-

    # Architecture
    kernel_size: int = 5
    stride: int = 1
    padding: int = 0
    # Paper (Falez 2020) uses 64 filters; threshold research requires more neurons
    # to make single-neuron perturbations statistically meaningful.
    num_filters: int = 256
    pool_size: int = 2

    # STDP
    stdp_variant: str = "multiplicative"
    stdp_lr: float = 0.1
    beta: float = 1.0
    w_min: float = 0.0
    w_max: float = 1.0

    # Threshold
    threshold_avg: float = 10.0
    threshold_std: float = 0.1
    min_threshold: float = 1.0
    threshold_lr: float = 1.0
    target_timestamp: float = 0.97

    # Training
    annealing: float = 0.95
    num_epochs: int = 100
    num_patches: int = 50
    patch_size: int = 5


MNIST = MnistHyperparams()
CIFAR10 = Cifar10Hyperparams()

_CONFIGS = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
}


def get_paper_hyperparams(dataset: str) -> dict:
    """Return paper-exact hyperparameters as a plain dict.

    :param dataset: 'mnist' or 'cifar10'.
    :returns: Dict of all hyperparameters for the dataset.
    :raises ValueError: If dataset name is unknown.
    """
    if dataset not in _CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. Available: {list(_CONFIGS.keys())}"
        )
    from dataclasses import asdict

    return asdict(_CONFIGS[dataset])
