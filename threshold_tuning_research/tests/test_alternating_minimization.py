import numpy as np

from threshold_tuning_research.refinement.alternating_minimization import (
    AMTrajectory,
    build_X,
    run_alternating_minimization,
)


def _separable_cache(seed=0):
    """Cache where the trained offset (index 3) is noise but offset 5 separates classes.

    A correct trust-region step should push neurons toward offset 5 and raise train acc
    well above the ~50% baseline.
    """
    rng = np.random.default_rng(seed)
    F, L, N, P = 6, 7, 80, 1
    target = 5
    y = np.array([0] * (N // 2) + [1] * (N // 2))
    sign = (y * 2 - 1).astype(np.float32)  # ±1
    cache = rng.normal(0, 0.3, (F, L, N, P)).astype(np.float32)
    for f in range(F):
        cache[f, target, :, 0] = 3.0 * sign + rng.normal(0, 0.3, N)
    fractions = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])  # zero at index 3
    return cache, y, fractions, target


def test_build_X_reconstructs_columns():
    cache = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)
    offsets = np.array([0, 1])
    X = build_X(cache, offsets)
    assert X.shape == (4, 2 * 2)  # N, F*P
    # neuron 0 at offset 0, neuron 1 at offset 1
    assert np.array_equal(X[:, 0:2], cache[0, 0])
    assert np.array_equal(X[:, 2:4], cache[1, 1])


def test_returns_trajectory_and_logs_history():
    cache, y, fractions, _ = _separable_cache()
    traj = run_alternating_minimization(cache, y, fractions, classifier="logistic", max_step=3, max_iter=8)
    assert isinstance(traj, AMTrajectory)
    assert len(traj.history) >= 1
    assert traj.final_offsets.shape == (cache.shape[0],)


def test_train_best_is_at_least_baseline():
    # Invariant behind ADR 0001: train-best is the max over iterations, incl. iter 0.
    cache, y, fractions, _ = _separable_cache()
    traj = run_alternating_minimization(cache, y, fractions, classifier="logistic", max_step=3, max_iter=8)
    assert traj.best_train >= traj.history[0]["train"] - 1e-9


def test_recovers_separable_offset():
    cache, y, fractions, target = _separable_cache()
    traj = run_alternating_minimization(cache, y, fractions, classifier="logistic", max_step=3, max_iter=8)
    # Baseline (all at the trained/zero offset, index 3) is noise → ~chance; AM should
    # improve a lot and move neurons off the baseline offset toward the separable region.
    assert traj.best_train > 0.8
    baseline_offset = 3
    moved = int((traj.train_best_offsets != baseline_offset).sum())
    assert moved >= cache.shape[0] // 2


def test_test_accuracy_optional():
    cache, y, fractions, _ = _separable_cache()
    traj = run_alternating_minimization(cache, y, fractions, classifier="logistic", max_step=3, max_iter=4)
    assert np.isnan(traj.history[0]["test"])  # no test cache provided → nan
