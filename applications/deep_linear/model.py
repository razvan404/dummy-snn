from spiking.layers import IntegrateAndFireMultilayer, SpikingSequential
from spiking.learning import Learner, STDP, WinnerTakesAll
from spiking.threshold import (
    NormalInitialization,
    CompetitiveThresholdAdaptation,
    PlasticityBalanceAdaptation,
)
from spiking.training import train
from torch.utils.data import DataLoader

ARCHITECTURE = [256, 128, 64]


def create_model(
    setup: dict,
    num_inputs: int,
    architecture: list[int] = ARCHITECTURE,
) -> IntegrateAndFireMultilayer:
    """Build a multi-layer integrate-and-fire model from a setup dict."""
    return IntegrateAndFireMultilayer(
        num_inputs=num_inputs,
        num_hidden=architecture[:-1],
        num_outputs=architecture[-1],
        threshold_initialization=NormalInitialization(**setup["threshold_init"]),
        refractory_period=float("inf"),
    )


def create_learner(
    model: IntegrateAndFireMultilayer,
    layer_idx: int,
    setup: dict,
) -> Learner:
    """Create a fresh Learner targeting a specific layer of the model."""
    return Learner(
        model.layers[layer_idx],
        learning_mechanism=STDP(**setup["stdp"]),
        competition=WinnerTakesAll(),
        threshold_adaptation=CompetitiveThresholdAdaptation(
            **setup["threshold_adaptation"]
        ),
    )


def train_layerwise(
    model: IntegrateAndFireMultilayer,
    setup: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    spike_shape: tuple[int, ...],
    num_layers: int = 1,
    num_epochs_per_layer: int = 50,
):
    """Train layers 0..num_layers-1 greedily, one at a time.

    Each layer trains with a sub-model containing only layers 0..layer_idx,
    so the forward pass does not propagate through untrained layers above.
    """
    for layer_idx in range(num_layers):
        sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
        learner = create_learner(model, layer_idx, setup)
        train(
            sub_model,
            learner,
            train_loader,
            num_epochs_per_layer,
            image_shape=spike_shape,
            val_loader=val_loader,
            progress=False,
        )


def apply_pba(
    model: IntegrateAndFireMultilayer,
    train_loader: DataLoader,
    spike_shape: tuple[int, ...],
    num_layers: int = 1,
    pba_kwargs: dict | None = None,
    num_epochs: int = 20,
):
    """Apply PlasticityBalanceAdaptation post-training to layers 0..num_layers-1."""
    if pba_kwargs is None:
        pba_kwargs = {}
    for layer_idx in range(num_layers):
        sub_model = SpikingSequential(*model.layers[: layer_idx + 1])
        adaptation = PlasticityBalanceAdaptation(**pba_kwargs)
        learner = Learner(
            model.layers[layer_idx],
            learning_mechanism=None,
            threshold_adaptation=adaptation,
        )
        train(
            sub_model,
            learner,
            train_loader,
            num_epochs,
            image_shape=spike_shape,
            early_stopping=False,
            progress=False,
        )
