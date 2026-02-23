from collections.abc import Callable

import tqdm
from torch.utils.data import DataLoader

from spiking.learning.learner import Learner
from spiking.spiking_module import SpikingModule
from .unsupervised_trainer import UnsupervisedTrainer


def train(
    model: SpikingModule,
    learner: Learner,
    train_loader: DataLoader,
    num_epochs: int,
    image_shape: tuple[int, ...],
    *,
    val_loader: DataLoader | None = None,
    on_batch_end: Callable[[int, float, str], None] | None = None,
    early_stopping: bool = True,
    progress: bool = True,
):
    trainer = UnsupervisedTrainer(
        model,
        learner,
        image_shape=image_shape,
        on_batch_end=on_batch_end,
        early_stopping=early_stopping,
    )
    epochs = tqdm.trange(num_epochs) if progress else range(num_epochs)
    for epoch in epochs:
        trainer.step_loader(train_loader, split="train")
        if val_loader:
            trainer.step_loader(val_loader, split="val")
        trainer.step_epoch()
