import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier

from spiking.evaluation.ridge_column_swap import RidgeColumnSwap


def _make_data(n_train=100, n_val=30, d=10, n_classes=3, seed=42):
    """Create synthetic classification data."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d).astype(np.float32)
    X_val = rng.randn(n_val, d).astype(np.float32)
    y_train = rng.randint(0, n_classes, n_train)
    y_val = rng.randint(0, n_classes, n_val)
    return X_train, X_val, y_train, y_val


class TestBaselinePrediction:
    """Verify baseline predictions match sklearn RidgeClassifier."""

    def test_matches_sklearn_multiclass(self):
        X_train, X_val, y_train, y_val = _make_data(n_classes=5)
        alpha = 1.0

        sklearn_clf = RidgeClassifier(alpha=alpha)
        sklearn_clf.fit(X_train, y_train)
        expected = sklearn_clf.predict(X_val)

        our_clf = RidgeColumnSwap(alpha=alpha)
        our_clf.fit(X_train, y_train)
        actual = our_clf.predict(X_val)

        np.testing.assert_array_equal(actual, expected)

    def test_matches_sklearn_binary(self):
        X_train, X_val, y_train, y_val = _make_data(n_classes=2)
        alpha = 0.5

        sklearn_clf = RidgeClassifier(alpha=alpha)
        sklearn_clf.fit(X_train, y_train)
        expected = sklearn_clf.predict(X_val)

        our_clf = RidgeColumnSwap(alpha=alpha)
        our_clf.fit(X_train, y_train)
        actual = our_clf.predict(X_val)

        np.testing.assert_array_equal(actual, expected)

    def test_different_alpha(self):
        X_train, X_val, y_train, _ = _make_data()
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            sklearn_clf = RidgeClassifier(alpha=alpha)
            sklearn_clf.fit(X_train, y_train)

            our_clf = RidgeColumnSwap(alpha=alpha)
            our_clf.fit(X_train, y_train)

            np.testing.assert_array_equal(
                our_clf.predict(X_val),
                sklearn_clf.predict(X_val),
                err_msg=f"Mismatch at alpha={alpha}",
            )


class TestColumnSwapCorrectness:
    """Verify predict_swapped matches full refit with modified data."""

    def test_single_column_swap(self):
        """Swapping one column should match a full refit."""
        X_train, X_val, y_train, _ = _make_data(d=10)
        rng = np.random.RandomState(99)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)

        # Woodbury prediction
        X_val_mod = X_val.copy()
        X_val_mod[:, col] = rng.randn(X_val.shape[0])
        pred_woodbury = clf.predict_swapped([col], new_col, X_val_mod)

        # Full refit
        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val_mod)

        np.testing.assert_array_equal(pred_woodbury, pred_refit)

    def test_multi_column_swap(self):
        """Swapping multiple columns should match a full refit."""
        X_train, X_val, y_train, _ = _make_data(d=16)
        rng = np.random.RandomState(99)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        cols = [2, 3, 4, 5]  # 4 columns (like conv pool_size=2)
        new_cols = rng.randn(X_train.shape[0], 4).astype(np.float32)

        X_val_mod = X_val.copy()
        X_val_mod[:, cols] = rng.randn(X_val.shape[0], 4)
        pred_woodbury = clf.predict_swapped(cols, new_cols, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val_mod)

        np.testing.assert_array_equal(pred_woodbury, pred_refit)

    def test_multiple_sequential_swaps_independent(self):
        """Each swap should be independent (not cumulative)."""
        X_train, X_val, y_train, _ = _make_data(d=10)
        rng = np.random.RandomState(77)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        # Swap column 0, then separately swap column 5 — both from baseline
        for col in [0, 5]:
            new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
            X_val_mod = X_val.copy()
            X_val_mod[:, col] = rng.randn(X_val.shape[0])
            pred_woodbury = clf.predict_swapped([col], new_col, X_val_mod)

            X_train_mod = X_train.copy()
            X_train_mod[:, col : col + 1] = new_col
            refit_clf = RidgeClassifier(alpha=1.0)
            refit_clf.fit(X_train_mod, y_train)
            pred_refit = refit_clf.predict(X_val_mod)

            np.testing.assert_array_equal(pred_woodbury, pred_refit)

    def test_no_change_gives_baseline(self):
        """Swapping a column with identical data should give baseline predictions."""
        X_train, X_val, y_train, _ = _make_data(d=8)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        baseline = clf.predict(X_val)
        col = 2
        same_col = X_train[:, col : col + 1].copy()
        swapped = clf.predict_swapped([col], same_col, X_val)

        np.testing.assert_array_equal(swapped, baseline)

    def test_different_alpha_values(self):
        """Woodbury should work correctly with different regularization."""
        X_train, X_val, y_train, _ = _make_data(d=10)
        rng = np.random.RandomState(123)
        col = 7
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, col] = rng.randn(X_val.shape[0])

        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = RidgeColumnSwap(alpha=alpha)
            clf.fit(X_train, y_train)
            pred_woodbury = clf.predict_swapped([col], new_col, X_val_mod)

            X_train_mod = X_train.copy()
            X_train_mod[:, col : col + 1] = new_col
            refit_clf = RidgeClassifier(alpha=alpha)
            refit_clf.fit(X_train_mod, y_train)
            pred_refit = refit_clf.predict(X_val_mod)

            np.testing.assert_array_equal(
                pred_woodbury,
                pred_refit,
                err_msg=f"Mismatch at alpha={alpha}",
            )

    def test_binary_classification_swap(self):
        """Column swap should work for binary classification."""
        X_train, X_val, y_train, _ = _make_data(d=8, n_classes=2)
        rng = np.random.RandomState(55)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        cols = [1, 2]
        new_cols = rng.randn(X_train.shape[0], 2).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, cols] = rng.randn(X_val.shape[0], 2)
        pred_woodbury = clf.predict_swapped(cols, new_cols, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val_mod)

        np.testing.assert_array_equal(pred_woodbury, pred_refit)


class TestApplySwap:
    """Test that apply_swap permanently updates classifier state."""

    def test_apply_then_predict_matches_refit(self):
        """After apply_swap, predict should match a fresh fit on modified data."""
        X_train, X_val, y_train, _ = _make_data(d=10)
        rng = np.random.RandomState(88)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)

        # Woodbury apply_swap then predict
        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col], new_col)
        pred_applied = clf.predict(X_val)

        # Full refit
        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val)

        np.testing.assert_array_equal(pred_applied, pred_refit)

    def test_chained_apply_swap(self):
        """Two sequential apply_swaps should match a fresh fit on both columns modified."""
        X_train, X_val, y_train, _ = _make_data(d=8)
        rng = np.random.RandomState(66)

        col_a, col_b = 2, 5
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        # Chained Woodbury
        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        clf.apply_swap([col_b], new_b)
        pred_chained = clf.predict(X_val)

        # Full refit with both columns modified
        X_train_mod = X_train.copy()
        X_train_mod[:, col_a : col_a + 1] = new_a
        X_train_mod[:, col_b : col_b + 1] = new_b
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val)

        np.testing.assert_array_equal(pred_chained, pred_refit)

    def test_apply_swap_then_predict_swapped(self):
        """After apply_swap on col A, predict_swapped on col B should match
        a fresh fit with both A and B modified."""
        X_train, X_val, y_train, _ = _make_data(d=10)
        rng = np.random.RandomState(77)

        col_a, col_b = 1, 6
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        # Apply A, then predict_swapped B
        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        X_val_mod = X_val.copy()
        X_val_mod[:, col_b] = rng.randn(X_val.shape[0])
        pred = clf.predict_swapped([col_b], new_b, X_val_mod)

        # Full refit with both
        X_train_mod = X_train.copy()
        X_train_mod[:, col_a : col_a + 1] = new_a
        X_train_mod[:, col_b : col_b + 1] = new_b
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val_mod)

        np.testing.assert_array_equal(pred, pred_refit)

    def test_multi_column_apply_swap(self):
        """apply_swap with multiple columns at once should match refit."""
        X_train, X_val, y_train, _ = _make_data(d=12)
        rng = np.random.RandomState(99)

        cols = [3, 4, 5, 6]
        new_cols = rng.randn(X_train.shape[0], 4).astype(np.float32)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)
        clf.apply_swap(cols, new_cols)
        pred_applied = clf.predict(X_val)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val)

        np.testing.assert_array_equal(pred_applied, pred_refit)


class TestLargerDimensions:
    """Test with dimensions closer to real conv scenarios."""

    def test_high_dimensional_features(self):
        """d=1024 (256 filters * 2*2 pool) — the real conv scenario."""
        rng = np.random.RandomState(42)
        d = 1024
        n_train, n_val = 200, 50
        X_train = rng.randn(n_train, d).astype(np.float32)
        X_val = rng.randn(n_val, d).astype(np.float32)
        y_train = rng.randint(0, 10, n_train)

        clf = RidgeColumnSwap(alpha=1.0)
        clf.fit(X_train, y_train)

        # Swap 4 columns (one filter with pool_size=2)
        cols = [100, 101, 102, 103]
        new_cols = rng.randn(n_train, 4).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, cols] = rng.randn(n_val, 4)

        pred_woodbury = clf.predict_swapped(cols, new_cols, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        refit_clf = RidgeClassifier(alpha=1.0)
        refit_clf.fit(X_train_mod, y_train)
        pred_refit = refit_clf.predict(X_val_mod)

        np.testing.assert_array_equal(pred_woodbury, pred_refit)
