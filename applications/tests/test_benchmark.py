import json
import os

import numpy as np
import pytest


class TestAggregateMetrics:
    def test_computes_mean_and_std_for_each_split_and_metric(self):
        from applications.common import aggregate_metrics

        all_metrics = [
            {
                "train": {
                    "accuracy": 0.90,
                    "precision": 0.88,
                    "recall": 0.87,
                    "f1": 0.87,
                },
                "validation": {
                    "accuracy": 0.80,
                    "precision": 0.78,
                    "recall": 0.77,
                    "f1": 0.77,
                },
            },
            {
                "train": {
                    "accuracy": 0.92,
                    "precision": 0.90,
                    "recall": 0.89,
                    "f1": 0.89,
                },
                "validation": {
                    "accuracy": 0.84,
                    "precision": 0.82,
                    "recall": 0.81,
                    "f1": 0.81,
                },
            },
        ]

        summary = aggregate_metrics(all_metrics)

        assert "train" in summary
        assert "validation" in summary
        for split in ("train", "validation"):
            for metric in ("accuracy", "precision", "recall", "f1"):
                assert "mean" in summary[split][metric]
                assert "std" in summary[split][metric]

        # Check train accuracy: mean of [0.90, 0.92] = 0.91
        assert summary["train"]["accuracy"]["mean"] == pytest.approx(0.91)
        assert summary["train"]["accuracy"]["std"] == pytest.approx(
            np.std([0.90, 0.92])
        )

        # Check validation f1: mean of [0.77, 0.81] = 0.79
        assert summary["validation"]["f1"]["mean"] == pytest.approx(0.79)
        assert summary["validation"]["f1"]["std"] == pytest.approx(np.std([0.77, 0.81]))

    def test_single_seed_has_zero_std(self):
        from applications.common import aggregate_metrics

        all_metrics = [
            {
                "train": {
                    "accuracy": 0.90,
                    "precision": 0.88,
                    "recall": 0.87,
                    "f1": 0.87,
                },
                "validation": {
                    "accuracy": 0.80,
                    "precision": 0.78,
                    "recall": 0.77,
                    "f1": 0.77,
                },
            },
        ]

        summary = aggregate_metrics(all_metrics)

        assert summary["train"]["accuracy"]["mean"] == pytest.approx(0.90)
        assert summary["train"]["accuracy"]["std"] == pytest.approx(0.0)


class TestMergeSeedResults:
    METRICS_A = {
        "train": {"accuracy": 0.90, "precision": 0.88, "recall": 0.87, "f1": 0.87},
        "validation": {"accuracy": 0.80, "precision": 0.78, "recall": 0.77, "f1": 0.77},
    }
    METRICS_B = {
        "train": {"accuracy": 0.92, "precision": 0.90, "recall": 0.89, "f1": 0.89},
        "validation": {"accuracy": 0.84, "precision": 0.82, "recall": 0.81, "f1": 0.81},
    }

    def _setup_seeds(self, tmp_path):
        for seed, metrics in [(1, self.METRICS_A), (2, self.METRICS_B)]:
            seed_dir = tmp_path / f"seed_{seed}"
            seed_dir.mkdir()
            with open(seed_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

    def test_creates_merged_results_file(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        merge_seed_results(str(tmp_path))
        assert (tmp_path / "merged_results.json").exists()

    def test_creates_summary_file(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        merge_seed_results(str(tmp_path))
        assert (tmp_path / "summary.json").exists()

    def test_merged_results_has_seeds_list(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        merge_seed_results(str(tmp_path))
        with open(tmp_path / "merged_results.json") as f:
            merged = json.load(f)
        assert merged["seeds"] == [1, 2]

    def test_merged_results_has_metric_lists(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        merge_seed_results(str(tmp_path))
        with open(tmp_path / "merged_results.json") as f:
            merged = json.load(f)
        assert merged["train"]["accuracy"] == [0.90, 0.92]
        assert merged["validation"]["f1"] == [0.77, 0.81]

    def test_summary_has_mean_and_std(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        merge_seed_results(str(tmp_path))
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        assert summary["train"]["accuracy"]["mean"] == pytest.approx(0.91)
        assert summary["validation"]["f1"]["std"] == pytest.approx(np.std([0.77, 0.81]))

    def test_ignores_non_seed_directories(self, tmp_path):
        from applications.common import merge_seed_results

        self._setup_seeds(tmp_path)
        other_dir = tmp_path / "other_stuff"
        other_dir.mkdir()
        merge_seed_results(str(tmp_path))
        with open(tmp_path / "merged_results.json") as f:
            merged = json.load(f)
        assert merged["seeds"] == [1, 2]

    def test_single_seed(self, tmp_path):
        from applications.common import merge_seed_results

        seed_dir = tmp_path / "seed_42"
        seed_dir.mkdir()
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(self.METRICS_A, f)
        merge_seed_results(str(tmp_path))
        with open(tmp_path / "merged_results.json") as f:
            merged = json.load(f)
        assert merged["seeds"] == [42]
        assert merged["train"]["accuracy"] == [0.90]
