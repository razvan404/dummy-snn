from typing import Literal

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC

from dataloaders import Dataloader
from spiking import SpikingModule, iterate_spikes
from spiking.evaluation.eval_utils import compute_metrics


class SpikingClassifierEvaluator:
    def __init__(
        self,
        model: SpikingModule,
        train_dataloader: Dataloader,
        validation_dataloader: Dataloader,
        shape: (int, int, int),
    ):
        self.shape = shape
        self.model = model
        self.X_train, self.y_train = self._dataloader_to_spike_times(train_dataloader)
        self.X_test, self.y_test = self._dataloader_to_spike_times(
            validation_dataloader
        )

    def _dataloader_to_spike_times(self, arg_dataloader: Dataloader):
        X, y = [], []
        for batch_idx, (spikes, label, _) in enumerate(
            arg_dataloader.iterate(batch_size=1), start=1
        ):
            for incoming_spikes, current_time, dt in iterate_spikes(
                spikes, shape=self.shape
            ):
                self.model.forward(
                    incoming_spikes.flatten(), current_time=current_time, dt=dt
                )
            X.append(torch.clamp(1.0 - self.model.spike_times, min=0, max=1.0).numpy())
            # TODO: take formula (10) into consideration for when the adaptation technique is the same one
            y.append(label)
            self.model.reset()
        return np.array(X), np.array(y)

    def plot_reduced_dataset(
        self, split: Literal["train", "val"], reducer: TransformerMixin | None = None
    ):
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        if not reducer:
            reducer = PCA(n_components=2)
            reducer.fit(X)
        X_pca = reducer.transform(X)

        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", alpha=0.7
        )
        plt.colorbar(scatter, label="Class Label")
        plt.title(f"{split.capitalize()} Data Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

        return reducer

    def _predict(
        self, classifier: ClassifierMixin | nn.Module, split: Literal["train", "val"]
    ):
        set = self.X_train if split == "train" else self.X_test
        if isinstance(classifier, nn.Module):
            classifier.eval()
            with torch.no_grad():
                X = torch.tensor(set).float()
                y_pred = classifier(X).argmax(dim=1).numpy()
        elif isinstance(classifier, ClassifierMixin):
            y_pred = classifier.predict(set)
        else:
            raise TypeError(
                "Classifier must be a PyTorch model or a Scikit-learn classifier."
            )
        return y_pred

    def eval_classifier(
        self,
        classifier: ClassifierMixin | nn.Module = None,
        train: bool = False,
        visualize: bool = True,
        verbose: bool = True,
    ):
        if classifier is None:
            classifier = LinearSVC(max_iter=20000)

        if train:
            if isinstance(classifier, ClassifierMixin):
                classifier.fit(self.X_train, self.y_train)
            else:
                raise TypeError(
                    "Classifier must be a Scikit-learn classifier for training."
                )

        if verbose:
            print("Train metrics:")
        y_pred = self._predict(classifier, split="train")
        train_metrics = compute_metrics(
            self.y_train, y_pred, visualize=visualize, verbose=verbose
        )

        if verbose:
            print("\nValidation metrics:")
        y_pred = self._predict(classifier, split="val")
        val_metrics = compute_metrics(
            self.y_test, y_pred, visualize=visualize, verbose=verbose
        )

        return train_metrics, val_metrics
