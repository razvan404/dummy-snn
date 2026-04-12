"""Benchmark CPU vs CUDA for RidgeColumnSwap and SVCColumnSwap.

Ridge: Woodbury updates — numpy (CPU) vs cupy (GPU).
SVC: full refit — sklearn (CPU) vs cuml (GPU).

Usage:
    python benchmarks/bench_classifiers.py
    python benchmarks/bench_classifiers.py --dims 512 1024 --n-train 50000
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BenchResult:
    name: str
    times: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return np.mean(self.times) * 1000

    @property
    def std_ms(self) -> float:
        return np.std(self.times) * 1000


def _make_data(
    n_train: int, n_val: int, d: int, n_classes: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, d).astype(np.float32)
    y_train = rng.randint(0, n_classes, n_train)
    X_val = rng.randn(n_val, d).astype(np.float32)
    y_val = rng.randint(0, n_classes, n_val)
    for c in range(n_classes):
        X_train[y_train == c, : min(n_classes, d)] += c * 1.5
        X_val[y_val == c, : min(n_classes, d)] += c * 1.5
    return X_train, X_val, y_train, y_val


def _sync_gpu():
    try:
        import cupy as _cp

        _cp.cuda.Device().synchronize()
    except Exception:
        pass


def bench_ridge(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    k: int,
    use_gpu: bool,
    n_repeats: int,
    n_warmup: int,
) -> dict[str, BenchResult]:
    from spiking.evaluation.ridge_column_swap import RidgeColumnSwap

    rng = np.random.RandomState(123)
    col_indices = list(range(0, k))
    new_cols = rng.randn(X_train.shape[0], k).astype(np.float32)
    X_val_mod = X_val.copy()
    X_val_mod[:, col_indices] = rng.randn(X_val.shape[0], k)

    results: dict[str, BenchResult] = {}
    tag = "gpu" if use_gpu else "cpu"

    # -- fit --
    r = BenchResult(f"Ridge.fit [{tag}]")
    for i in range(n_warmup + n_repeats):
        clf = RidgeColumnSwap(alpha=1.0, use_gpu=use_gpu)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        if use_gpu:
            _sync_gpu()
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["fit"] = r

    clf = RidgeColumnSwap(alpha=1.0, use_gpu=use_gpu)
    clf.fit(X_train, y_train)
    if use_gpu:
        _sync_gpu()

    # -- predict --
    r = BenchResult(f"Ridge.predict [{tag}]")
    for i in range(n_warmup + n_repeats):
        t0 = time.perf_counter()
        clf.predict(X_val)
        if use_gpu:
            _sync_gpu()
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["predict"] = r

    # -- predict_swapped --
    r = BenchResult(f"Ridge.predict_swapped [{tag}]")
    for i in range(n_warmup + n_repeats):
        t0 = time.perf_counter()
        clf.predict_swapped(col_indices, new_cols, X_val_mod)
        if use_gpu:
            _sync_gpu()
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["predict_swapped"] = r

    # -- apply_swap --
    r = BenchResult(f"Ridge.apply_swap [{tag}]")
    for i in range(n_warmup + n_repeats):
        clf_copy = RidgeColumnSwap(alpha=1.0, use_gpu=use_gpu)
        clf_copy.fit(X_train, y_train)
        if use_gpu:
            _sync_gpu()
        new_cols_i = rng.randn(X_train.shape[0], k).astype(np.float32)
        t0 = time.perf_counter()
        clf_copy.apply_swap(col_indices, new_cols_i)
        if use_gpu:
            _sync_gpu()
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["apply_swap"] = r

    return results


def bench_svc(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    k: int,
    use_gpu: bool,
    n_repeats: int,
    n_warmup: int,
) -> dict[str, BenchResult]:
    from spiking.evaluation.svc_column_swap import SVCColumnSwap

    rng = np.random.RandomState(456)
    col_indices = list(range(0, k))
    new_cols = rng.randn(X_train.shape[0], k).astype(np.float32)
    X_val_mod = X_val.copy()
    X_val_mod[:, col_indices] = rng.randn(X_val.shape[0], k)

    results: dict[str, BenchResult] = {}
    tag = "gpu" if use_gpu else "cpu"

    # -- fit --
    r = BenchResult(f"SVC.fit [{tag}]")
    for i in range(n_warmup + n_repeats):
        clf = SVCColumnSwap(use_gpu=use_gpu, random_state=0)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["fit"] = r

    clf = SVCColumnSwap(use_gpu=use_gpu, random_state=0)
    clf.fit(X_train, y_train)

    # -- predict --
    r = BenchResult(f"SVC.predict [{tag}]")
    for i in range(n_warmup + n_repeats):
        t0 = time.perf_counter()
        clf.predict(X_val)
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["predict"] = r

    # -- predict_swapped (= refit + predict) --
    r = BenchResult(f"SVC.predict_swapped [{tag}]")
    for i in range(n_warmup + n_repeats):
        t0 = time.perf_counter()
        clf.predict_swapped(col_indices, new_cols, X_val_mod)
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["predict_swapped"] = r

    # -- apply_swap (= refit in-place) --
    r = BenchResult(f"SVC.apply_swap [{tag}]")
    for i in range(n_warmup + n_repeats):
        clf_copy = SVCColumnSwap(use_gpu=use_gpu, random_state=0)
        clf_copy.fit(X_train, y_train)
        new_cols_i = rng.randn(X_train.shape[0], k).astype(np.float32)
        t0 = time.perf_counter()
        clf_copy.apply_swap(col_indices, new_cols_i)
        if i >= n_warmup:
            r.times.append(time.perf_counter() - t0)
    results["apply_swap"] = r

    return results


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cuml() -> bool:
    try:
        import cuml  # noqa: F401

        return True
    except ImportError:
        return False


def _print_table(
    cpu_results: dict[str, BenchResult],
    gpu_results: dict[str, BenchResult] | None,
) -> None:
    ops = list(cpu_results.keys())
    if gpu_results:
        header = f"{'Operation':<30} {'CPU (ms)':>14} {'GPU (ms)':>14} {'Speedup':>10}"
        print(header)
        print("-" * len(header))
        for op in ops:
            cpu = cpu_results[op]
            gpu = gpu_results[op]
            speedup = cpu.mean_ms / gpu.mean_ms if gpu.mean_ms > 0 else float("inf")
            print(
                f"  {cpu.name:<28} {cpu.mean_ms:>8.1f} ± {cpu.std_ms:<5.1f}"
                f"{gpu.mean_ms:>8.1f} ± {gpu.std_ms:<5.1f}"
                f"{speedup:>8.1f}x"
            )
    else:
        header = f"{'Operation':<30} {'Time (ms)':>16}"
        print(header)
        print("-" * len(header))
        for op in ops:
            r = cpu_results[op]
            print(f"  {r.name:<28} {r.mean_ms:>8.1f} ± {r.std_ms:.1f}")


def run_benchmark(
    dims: list[int],
    n_train: int,
    n_val: int,
    n_classes: int,
    pool_size: int,
    n_repeats: int,
    n_warmup: int,
) -> None:
    has_gpu_ridge = _has_cupy()
    has_gpu_svc = _has_cuml()
    k = pool_size * pool_size

    print(
        f"Config: n_train={n_train}, n_val={n_val}, n_classes={n_classes}, "
        f"k={k} (pool {pool_size}x{pool_size})"
    )
    print(f"Repeats: {n_repeats} (+ {n_warmup} warmup)")
    print(f"GPU Ridge (cupy): {'available' if has_gpu_ridge else 'not available'}")
    print(f"GPU SVC   (cuml): {'available' if has_gpu_svc else 'not available'}")
    print()

    for d in dims:
        n_filters = d // k
        print(f"{'=' * 68}")
        print(f"d={d} ({n_filters} filters x {pool_size}x{pool_size} pool)")
        print(f"{'=' * 68}")

        X_train, X_val, y_train, y_val = _make_data(n_train, n_val, d, n_classes)

        # --- Ridge ---
        print("\n  Ridge (Woodbury):")
        cpu_ridge = bench_ridge(X_train, X_val, y_train, k, False, n_repeats, n_warmup)
        gpu_ridge = None
        if has_gpu_ridge:
            gpu_ridge = bench_ridge(
                X_train, X_val, y_train, k, True, n_repeats, n_warmup
            )
        _print_table(cpu_ridge, gpu_ridge)

        # --- SVC ---
        print("\n  LinearSVC (refit):")
        cpu_svc = bench_svc(X_train, X_val, y_train, k, False, n_repeats, n_warmup)
        gpu_svc = None
        if has_gpu_svc:
            gpu_svc = bench_svc(X_train, X_val, y_train, k, True, n_repeats, n_warmup)
        _print_table(cpu_svc, gpu_svc)

        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark classifier CPU vs CUDA")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[512, 1024],
        help="Feature dimensions to benchmark (default: 512 1024)",
    )
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=1)
    args = parser.parse_args()

    run_benchmark(
        dims=args.dims,
        n_train=args.n_train,
        n_val=args.n_val,
        n_classes=args.n_classes,
        pool_size=args.pool_size,
        n_repeats=args.n_repeats,
        n_warmup=args.n_warmup,
    )


if __name__ == "__main__":
    main()
