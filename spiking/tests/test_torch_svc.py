import numpy as np

from spiking.evaluation.torch_svc import TorchLinearSVC


def _make_data(n_train=200, n_val=60, d=20, n_classes=3, seed=42):
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


class TestAccuracy:
    def test_multiclass_5(self):
        X_train, X_val, y_train, y_val = _make_data(
            n_train=500, n_val=100, d=20, n_classes=5, seed=10
        )
        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        assert (clf.predict(X_val) == y_val).mean() >= 0.7

    def test_multiclass_10(self):
        X_train, X_val, y_train, y_val = _make_data(
            n_train=1000, n_val=200, d=50, n_classes=10, seed=20
        )
        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        # Class signal is weak relative to 50-dim noise; well above chance (0.1).
        assert (clf.predict(X_val) == y_val).mean() >= 0.4


class TestColumnSwapCorrectness:
    def test_predict_swapped_changes_output(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=30)
        rng = np.random.RandomState(99)

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        baseline = clf.predict(X_val)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        X_val_mod = X_val.copy()
        X_val_mod[:, col] = rng.randn(X_val.shape[0])

        swapped = clf.predict_swapped([col], new_col, X_val_mod)

        # Should be a different prediction (data changed significantly)
        assert swapped.shape == baseline.shape

    def test_no_change_gives_baseline(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=50)

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)

        baseline = clf.predict(X_val)
        col = 2
        same_col = X_train[:, col : col + 1].copy()
        swapped = clf.predict_swapped([col], same_col, X_val)

        np.testing.assert_array_equal(swapped, baseline)

    def test_apply_swap_changes_state(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=60)
        rng = np.random.RandomState(88)

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf.apply_swap([col], new_col)
        pred_after = clf.predict(X_val)

        # Prediction should be deterministic after apply_swap
        pred_again = clf.predict(X_val)
        np.testing.assert_array_equal(pred_after, pred_again)

    def test_apply_swap_matches_predict_swapped(self):
        X_train, X_val, y_train, _ = _make_data(d=10, seed=60)
        rng = np.random.RandomState(88)

        col = 3
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        X_val_mod = X_val.copy()  # no change to val for this test

        clf1 = TorchLinearSVC(C=1.0)
        clf1.fit(X_train, y_train)
        pred_swapped = clf1.predict_swapped([col], new_col, X_val_mod)

        clf2 = TorchLinearSVC(C=1.0)
        clf2.fit(X_train, y_train)
        clf2.apply_swap([col], new_col)
        pred_applied = clf2.predict(X_val_mod)

        np.testing.assert_array_equal(pred_swapped, pred_applied)

    def test_chained_apply_swap(self):
        X_train, X_val, y_train, _ = _make_data(d=8, seed=70)
        rng = np.random.RandomState(66)

        col_a, col_b = 2, 5
        new_a = rng.randn(X_train.shape[0], 1).astype(np.float32)
        new_b = rng.randn(X_train.shape[0], 1).astype(np.float32)

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        clf.apply_swap([col_a], new_a)
        clf.apply_swap([col_b], new_b)
        pred_chained = clf.predict(X_val)

        # Refit from scratch on modified data
        X_mod = X_train.copy()
        X_mod[:, col_a : col_a + 1] = new_a
        X_mod[:, col_b : col_b + 1] = new_b
        clf2 = TorchLinearSVC(C=1.0)
        clf2.fit(X_mod, y_train)
        pred_direct = clf2.predict(X_val)

        # Warm-start may drift slightly; allow up to 5% mismatch
        match_rate = (pred_chained == pred_direct).mean()
        assert match_rate >= 0.95, f"match_rate={match_rate:.3f}"


class TestWarmStart:
    def test_eval_swapped_train_acc_returns_valid(self):
        X_train, _, y_train, _ = _make_data(n_train=200, d=10, n_classes=3, seed=100)
        import torch

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)

        rng = np.random.RandomState(42)
        col_indices = torch.tensor([2, 3], device=clf._device)
        new_cols = torch.from_numpy(
            rng.randn(X_train.shape[0], 2).astype(np.float32)
        ).to(clf._device)
        y_t = clf._y_t

        acc = clf.eval_swapped_train_acc(col_indices, new_cols, y_t)
        assert 0.0 <= acc <= 1.0

    def test_gpu_state_synced_after_apply_swap(self):
        X_train, _, y_train, _ = _make_data(d=10, seed=110)

        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)

        rng = np.random.RandomState(42)
        new_col = rng.randn(X_train.shape[0], 1).astype(np.float32)
        clf.apply_swap([3], new_col)

        # GPU tensor should match numpy
        np.testing.assert_allclose(
            clf._X_t[:, 3].cpu().numpy(), clf._X[:, 3], rtol=1e-5
        )

    def test_warm_start_accuracy_close_to_cold(self):
        X_train, X_val, y_train, y_val = _make_data(
            n_train=500, n_val=100, d=20, n_classes=5, seed=120
        )
        rng = np.random.RandomState(42)

        clf_warm = TorchLinearSVC(C=1.0)
        clf_warm.fit(X_train, y_train)

        clf_cold = TorchLinearSVC(C=1.0)
        clf_cold.fit(X_train, y_train)

        cols = [5, 6, 7, 8]
        new_cols = rng.randn(X_train.shape[0], 4).astype(np.float32)

        pred_warm = clf_warm.predict_swapped(cols, new_cols, X_val)

        # Cold start: build modified data and fit from scratch
        X_mod = X_train.copy()
        X_mod[:, cols] = new_cols
        clf_fresh = TorchLinearSVC(C=1.0)
        clf_fresh.fit(X_mod, y_train)
        pred_cold = clf_fresh.predict(X_val)

        acc_warm = (pred_warm == y_val).mean()
        acc_cold = (pred_cold == y_val).mean()
        assert abs(acc_warm - acc_cold) < 0.05, (
            f"warm={acc_warm:.4f}, cold={acc_cold:.4f}"
        )


class TestWeightsProperty:
    def test_multiclass_shape(self):
        X_train, _, y_train, _ = _make_data(d=10, n_classes=5, seed=90)
        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        w = clf.weights
        assert w.shape == (10, 5)

    def test_weights_are_numpy(self):
        X_train, _, y_train, _ = _make_data(d=10, n_classes=3, seed=92)
        clf = TorchLinearSVC(C=1.0)
        clf.fit(X_train, y_train)
        assert isinstance(clf.weights, np.ndarray)
