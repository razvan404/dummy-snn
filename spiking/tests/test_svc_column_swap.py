import numpy as np
from sklearn.svm import LinearSVC

from spiking.evaluation.svc_column_swap import SVCColumnSwap


def _make_data(n_train=100, n_val=30, d=10, n_classes=3, seed=42):
    """Create synthetic classification data with separable clusters."""
    rng = np.random.RandomState(seed)
    # Use class-correlated features so LinearSVC converges reliably
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
    """Verify baseline predictions match sklearn LinearSVC."""

    def test_matches_sklearn_multiclass(self):
        X_train, X_val, y_train, _ = _make_data(n_classes=5, seed=10)

        sk = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        sk.fit(X_train, y_train)
        expected = sk.predict(X_val)

        ours = SVCColumnSwap(random_state=0)
        ours.fit(X_train, y_train)
        actual = ours.predict(X_val)

        np.testing.assert_array_equal(actual, expected)

    def test_matches_sklearn_binary(self):
        X_train, X_val, y_train, _ = _make_data(n_classes=2, seed=20)

        sk = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        sk.fit(X_train, y_train)
        expected = sk.predict(X_val)

        ours = SVCColumnSwap(random_state=0)
        ours.fit(X_train, y_train)
        actual = ours.predict(X_val)

        np.testing.assert_array_equal(actual, expected)


class TestColumnSwapCorrectness:
    """Verify predict_swapped matches full refit with modified data."""

    def test_single_column_swap(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=30)
        rng = np.random.RandomState(99)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, col] = rng.randn(X_val.shape[0])

        pred_swapped = clf.predict_swapped([col], new_col, X_val_mod)

        # Full refit
        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        refit = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        refit.fit(X_train_mod.astype(np.float64), y_train)
        pred_refit = refit.predict(X_val_mod)

        np.testing.assert_array_equal(pred_swapped, pred_refit)

    def test_multi_column_swap(self):
        X_train, X_val, y_train, _ = _make_data(d=16, seed=40)
        rng = np.random.RandomState(99)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)

        cols = [2, 3, 4, 5]
        new_cols = rng.randn(X_train.shape[0], 4).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, cols] = rng.randn(X_val.shape[0], 4)

        pred_swapped = clf.predict_swapped(cols, new_cols, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, cols] = new_cols
        refit = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        refit.fit(X_train_mod.astype(np.float64), y_train)
        pred_refit = refit.predict(X_val_mod)

        np.testing.assert_array_equal(pred_swapped, pred_refit)

    def test_no_change_gives_baseline(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=50)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)

        baseline = clf.predict(X_val)
        col = 2
        same_col = X_train[:, col : col + 1].copy()
        swapped = clf.predict_swapped([col], same_col, X_val)

        np.testing.assert_array_equal(swapped, baseline)


class TestApplySwap:
    """Test that apply_swap permanently updates classifier state."""

    def test_apply_then_predict_matches_refit(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=60)
        rng = np.random.RandomState(88)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col], new_col)
        pred_applied = clf.predict(X_val)

        X_train_mod = X_train.copy()
        X_train_mod[:, col : col + 1] = new_col
        refit = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        refit.fit(X_train_mod.astype(np.float64), y_train)
        pred_refit = refit.predict(X_val)

        np.testing.assert_array_equal(pred_applied, pred_refit)

    def test_chained_apply_swap(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=70)
        rng = np.random.RandomState(66)

        col_a, col_b = 2, 5
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        clf.apply_swap([col_b], new_b)
        pred_chained = clf.predict(X_val)

        X_train_mod = X_train.copy()
        X_train_mod[:, col_a : col_a + 1] = new_a
        X_train_mod[:, col_b : col_b + 1] = new_b
        refit = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        refit.fit(X_train_mod.astype(np.float64), y_train)
        pred_refit = refit.predict(X_val)

        np.testing.assert_array_equal(pred_chained, pred_refit)

    def test_apply_swap_then_predict_swapped(self):
        """apply_swap on A, then predict_swapped on B = refit with both."""
        X_train, X_val, y_train, _ = _make_data(d=10, seed=80)
        rng = np.random.RandomState(77)

        col_a, col_b = 1, 6
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        X_val_mod = X_val.copy()
        X_val_mod[:, col_b] = rng.randn(X_val.shape[0])
        pred = clf.predict_swapped([col_b], new_b, X_val_mod)

        X_train_mod = X_train.copy()
        X_train_mod[:, col_a : col_a + 1] = new_a
        X_train_mod[:, col_b : col_b + 1] = new_b
        refit = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        refit.fit(X_train_mod.astype(np.float64), y_train)
        pred_refit = refit.predict(X_val_mod)

        np.testing.assert_array_equal(pred, pred_refit)


class TestWeightsProperty:
    """Test the weights property returns correct shape and values."""

    def test_multiclass_shape(self):
        X_train, _, y_train, _ = _make_data(d=10, n_classes=5, seed=90)
        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)
        w = clf.weights
        assert w.shape == (10, 5)  # (d, K)

    def test_binary_shape(self):
        X_train, _, y_train, _ = _make_data(d=8, n_classes=2, seed=91)
        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)
        w = clf.weights
        assert w.shape == (8, 1)  # (d, 1) — transposed from sklearn's (1, d)

    def test_weights_match_sklearn_coef(self):
        X_train, _, y_train, _ = _make_data(d=10, n_classes=3, seed=92)
        clf = SVCColumnSwap(random_state=0)
        clf.fit(X_train, y_train)

        sk = LinearSVC(dual=False, tol=1e-3, max_iter=10000, random_state=0)
        sk.fit(X_train.astype(np.float64), y_train)

        np.testing.assert_array_almost_equal(clf.weights, sk.coef_.T)
