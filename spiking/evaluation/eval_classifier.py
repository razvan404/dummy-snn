import numpy as np
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from spiking.evaluation.eval_utils import compute_metrics


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin | None = None,
) -> tuple[dict, dict]:
    """Fit a classifier and return (train_metrics, val_metrics).

    Defaults to LinearSVC if no classifier is provided.
    """
    if classifier is None:
        classifier = LinearSVC(dual=False, tol=1e-3, max_iter=10000)

    classifier.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, classifier.predict(X_train))
    val_metrics = compute_metrics(y_test, classifier.predict(X_test))

    return train_metrics, val_metrics


def plot_reduced_features(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    reducer: TransformerMixin | None = None,
) -> TransformerMixin:
    """Plot PCA-reduced features on the current matplotlib axes.

    Returns the fitted reducer for reuse on other splits.
    """
    import matplotlib.pyplot as plt

    if reducer is None:
        reducer = PCA(n_components=2)
        reducer.fit(X)
    X_reduced = reducer.transform(X)

    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="viridis", edgecolor="k", alpha=0.7
    )
    plt.colorbar(scatter, label="Class Label")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()

    return reducer
