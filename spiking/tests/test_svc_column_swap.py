"""SVCColumnSwap is a thin facade over TorchLinearSVC; we test API & contracts here."""

import numpy as np

from spiking.evaluation.svc_column_swap import SVCColumnSwap


def _make_data(n_train=200, n_val=60, d=10, n_classes=3, seed=42):
    """Synthetic separable clusters — used across cases."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d).astype(np.float32)
    y_train = rng.randint(0, n_classes, n_train)
    for c in range(n_classes):
        X_train[y_train == c, :n_classes] += c * 2.0

    X_val = rng.randn(n_val, d).astype(np.float32)
    y_val = rng.randint(0, n_classes, n_val)
    for c in range(n_classes):
        X_val[y_val == c, :n_classes] += c * 2.0
    return X_train, X_val, y_train, y_val


class TestBaselinePrediction:
    """Baseline fit should reach reasonable accuracy on separable clusters."""

    def test_multiclass_accuracy(self):
        X_train, X_val, y_train, y_val = _make_data(n_classes=5, seed=10)
        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        assert (clf.predict(X_val) == y_val).mean() >= 0.7

    def test_binary_accuracy(self):
        X_train, X_val, y_train, y_val = _make_data(n_classes=2, seed=20)
        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        assert (clf.predict(X_val) == y_val).mean() >= 0.8


class TestColumnSwapCorrectness:
    """``predict_swapped`` should match a fresh fit on the modified matrix."""

    def test_single_column_swap_matches_fresh(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=30)
        rng = np.random.RandomState(99)

        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, col] = rng.randn(X_val.shape[0])

        pred_swapped = clf.predict_swapped([col], new_col, X_val_mod)

        # Reference: fresh classifier on modified data.
        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        ref = SVCColumnSwap()
        ref.fit(X_train_mod, y_train)
        pred_ref = ref.predict(X_val_mod)

        # Warm-start is an approximation; allow small disagreement.
        disagree = (pred_swapped != pred_ref).mean()
        assert disagree <= 0.10, f"too many disagreements: {disagree:.3f}"

    def test_multi_column_swap_matches_fresh(self):
        X_train, X_val, y_train, _ = _make_data(d=16, seed=40)
        rng = np.random.RandomState(99)

        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)

        cols = [2, 3, 4, 5]
        new_cols = rng.randn(X_train.shape[0], 4).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, cols] = rng.randn(X_val.shape[0], 4)

        pred_swapped = clf.predict_swapped(cols, new_cols, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        ref = SVCColumnSwap()
        ref.fit(X_train_mod, y_train)
        pred_ref = ref.predict(X_val_mod)

        disagree = (pred_swapped != pred_ref).mean()
        assert disagree <= 0.10, f"too many disagreements: {disagree:.3f}"

    def test_no_change_gives_baseline(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=50)
        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)

        baseline = clf.predict(X_val)
        col = 2
        same_col = X_train[:, col : col + 1].copy()
        swapped = clf.predict_swapped([col], same_col, X_val)

        np.testing.assert_array_equal(swapped, baseline)


class TestApplySwap:
    """``apply_swap`` permanently updates the classifier state."""

    def test_apply_then_predict_matches_fresh(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=60)
        rng = np.random.RandomState(88)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        clf.apply_swap([col], new_col)
        pred_applied = clf.predict(X_val)

        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        ref = SVCColumnSwap()
        ref.fit(X_train_mod, y_train)
        pred_ref = ref.predict(X_val)

        disagree = (pred_applied != pred_ref).mean()
        assert disagree <= 0.10

    def test_chained_apply_swap(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=70)
        rng = np.random.RandomState(66)

        col_a, col_b = 2, 5
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        clf.apply_swap([col_b], new_b)
        pred_chained = clf.predict(X_val)

        X_train_mod = X_train.copy()
        X_train_mod[:, col_a : col_a + 1] = new_a
        X_train_mod[:, col_b : col_b + 1] = new_b
        ref = SVCColumnSwap()
        ref.fit(X_train_mod, y_train)
        pred_ref = ref.predict(X_val)

        disagree = (pred_chained != pred_ref).mean()
        assert disagree <= 0.15


class TestWeightsProperty:
    """``weights`` returns ``(d, K)``."""

    def test_multiclass_shape(self):
        X_train, _, y_train, _ = _make_data(d=10, n_classes=5, seed=90)
        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        w = clf.weights
        assert w.shape == (10, 5)

    def test_binary_shape(self):
        X_train, _, y_train, _ = _make_data(d=8, n_classes=2, seed=91)
        clf = SVCColumnSwap()
        clf.fit(X_train, y_train)
        w = clf.weights
        # TorchLinearSVC uses one column per class even for binary.
        assert w.shape == (8, 2)
