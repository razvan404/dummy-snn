import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_real: np.ndarray, y_pred: np.ndarray, visualize: bool):
    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, average="macro")
    recall = recall_score(y_real, y_pred, average="macro")
    f1 = f1_score(y_real, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_real, y_pred)

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
