import numpy as np
import pytest

from applications.pipeline import FeatureCache, LayerSpec, RunSpec, feature_cache_filename
from applications.pipeline.datasets import dataset_names, get, load_split_data


class TestRunSpecLayout:
    def test_sweep_segment_uniform_across_datasets(self):
        # The Fashion `sweep` bug was a layout disagreement; every dataset must use sweep.
        assert "sweep" in str(RunSpec.single("cifar10", 256, 0.70, 1).model_dir)
        assert "sweep" in str(RunSpec.single("fashion_mnist", 256, 0.85, 1).model_dir)

    def test_single_layer_paths_are_backward_compatible(self):
        assert str(RunSpec.single("cifar10", 256, 0.70, 1).model_dir) == (
            "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_1"
        )
        assert str(RunSpec.single("fashion_mnist", 256, 0.85, 3).model_dir) == (
            "logs/fashion_mnist/sweep/nf_256/tobj_0.85/seed_3"
        )
        assert RunSpec.single("cifar10", 256, 0.70, 1).prefix_dir is None

    def test_two_layer_nests_under_l1_dir(self):
        run = RunSpec("cifar10", 1, layers=(LayerSpec(256, 0.70), LayerSpec(128, 0.85)))
        assert str(run.model_dir) == (
            "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_1/L2_nf128_tobj0.85"
        )
        # prefix is the fixed single-layer L1 dir (shared across the L2 sweep)
        assert str(run.prefix_dir) == "logs/cifar10_whitened/sweep/nf_256/tobj_0.70/seed_1"
        assert run.target == LayerSpec(128, 0.85)

    def test_tobj_must_strictly_increase(self):
        with pytest.raises(ValueError):
            RunSpec("cifar10", 1, layers=(LayerSpec(256, 0.85), LayerSpec(128, 0.70)))
        # Equal tobjs are allowed (non-decreasing)
        RunSpec("cifar10", 1, layers=(LayerSpec(256, 0.70), LayerSpec(128, 0.70)))

    def test_cache_path_matches_filename_helper(self):
        spec = RunSpec.single("cifar10", 256, 0.70, 1)
        assert spec.cache_path(0.05, 0.75).name == feature_cache_filename(0.05, 0.75)
        assert spec.cache_path(0.05, 0.75).parent == spec.model_dir

    def test_refinement_dir(self):
        d = RunSpec.single("cifar10", 256, 0.70, 2).refinement_dir(
            "alternating_minimization", "logistic_linear_step2"
        )
        assert str(d) == (
            "logs/snn_weight_analysis/alternating_minimization/logistic_linear_step2/cifar10/seed_2"
        )


class TestFeatureCache:
    def _make(self):
        F, L, N, P = 3, 5, 8, 4
        return FeatureCache(
            train_cache=np.random.rand(F, L, N, P).astype(np.float32),
            test_cache=np.random.rand(F, L, N, P).astype(np.float32),
            y_train=np.random.randint(0, 10, N),
            y_test=np.random.randint(0, 10, N),
            original_thresholds=np.random.rand(F).astype(np.float32),
            perturbation_fractions=[-0.10, -0.05, 0.0, 0.05, 0.10],
            step_size=0.05,
            max_drift=0.10,
            pool_size=2,
            t_target=0.70,
        )

    def test_roundtrip_preserves_schema(self, tmp_path):
        fc = self._make()
        path = tmp_path / "fc.pt"
        fc.save(path)
        loaded = FeatureCache.load(path)
        assert np.array_equal(loaded.train_cache, fc.train_cache)
        assert loaded.perturbation_fractions == fc.perturbation_fractions
        assert loaded.t_target == fc.t_target
        assert loaded.pool_size == fc.pool_size

    def test_zero_index_is_trained_offset(self):
        assert self._make().zero_index == 2  # 0.0 sits at index 2 of the 5 offsets


class TestDatasetRegistry:
    def test_core_datasets_registered(self):
        names = dataset_names()
        assert "cifar10" in names and "fashion_mnist" in names

    def test_hyperparams_key_maps(self):
        assert get("cifar10").hyperparams_key == "cifar10"

    def test_unknown_dataset_raises(self):
        try:
            load_split_data("does_not_exist")
            assert False, "expected ValueError"
        except ValueError:
            pass
