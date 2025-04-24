from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.svm import SVC

from dataloaders import Dataloader
from spiking import SpikingModule, iterate_spikes


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

    def plot_reduced_dataset(self, split: Literal["train", "val"]):
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", alpha=0.7
        )
        plt.colorbar(scatter, label="Class Label")
        plt.title(f"{split.capitalize()} Data Visualized with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

    def train_classifier_and_compute_metrics(
        self, classifier=None, visualize: bool = False
    ):
        if classifier is None:
            classifier = SVC()

        classifier.fit(self.X_train, self.y_train)

        print(
            "Train accuracy:",
            accuracy_score(self.y_train, classifier.predict(self.X_train)),
        )

        y_pred = classifier.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        recall = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        if visualize:
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix")
            plt.show()

        return accuracy, precision, recall, f1
