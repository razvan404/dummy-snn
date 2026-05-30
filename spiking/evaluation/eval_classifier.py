import numpy as np

from spiking.evaluation.eval_utils import compute_metrics
from spiking.evaluation.torch_svc import TorchLinearSVC


def evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier=None,
) -> tuple[dict, dict]:
    if classifier is None:
        classifier = TorchLinearSVC(C=1.0)

    classifier.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, np.asarray(classifier.predict(X_train)))
    val_metrics = compute_metrics(y_test, np.asarray(classifier.predict(X_test)))

    return train_metrics, val_metrics
