import json
import os

import pytest

from applications.search import resolve_search_space, find_hyperparameters


class TestResolveSearchSpace:
    def test_single_tunable_param(self):
        search_space = {
            "stdp": {
                "tau_pre": [0.05, 0.1],
                "tau_post": 0.1,
            },
        }
        param_names, configs = resolve_search_space(search_space)

        assert len(configs) == 2
        assert ("stdp", "tau_pre") in param_names

    def test_multiple_tunable_params_across_components(self):
        search_space = {
            "stdp": {
                "tau_pre": [0.05, 0.1],
                "learning_rate": [0.01, 0.1],
                "tau_post": 0.1,
            },
            "threshold_adaptation": {
                "min_threshold": 1.0,
                "learning_rate": [1.0, 5.0, 10.0],
            },
        }
        param_names, configs = resolve_search_space(search_space)

        # 2 * 2 * 3 = 12 combinations
        assert len(configs) == 12
        assert len(param_names) == 3
        assert ("stdp", "tau_pre") in param_names
        assert ("stdp", "learning_rate") in param_names
        assert ("threshold_adaptation", "learning_rate") in param_names

    def test_no_tunable_params_returns_single_config(self):
        search_space = {
            "stdp": {
                "tau_pre": 0.1,
                "tau_post": 0.1,
            },
        }
        param_names, configs = resolve_search_space(search_space)

        assert len(configs) == 1
        assert len(param_names) == 0
        assert configs[0] == search_space

    def test_resolved_configs_have_no_lists(self):
        search_space = {
            "stdp": {
                "tau_pre": [0.05, 0.1],
                "learning_rate": [0.01, 0.1],
                "tau_post": 0.1,
            },
        }
        _, configs = resolve_search_space(search_space)

        for config in configs:
            for key, kwargs in config.items():
                for param_value in kwargs.values():
                    assert not isinstance(
                        param_value, list
                    ), f"Config still has list for {key}: {param_value}"

    def test_resolved_configs_have_correct_values(self):
        search_space = {
            "stdp": {
                "tau_pre": [0.05, 0.1],
                "tau_post": 0.5,
            },
        }
        _, configs = resolve_search_space(search_space)

        tau_pre_values = {c["stdp"]["tau_pre"] for c in configs}
        assert tau_pre_values == {0.05, 0.1}

        # Fixed param should be the same in all configs
        for c in configs:
            assert c["stdp"]["tau_post"] == 0.5

    def test_does_not_mutate_original_search_space(self):
        search_space = {
            "stdp": {
                "tau_pre": [0.05, 0.1],
                "tau_post": 0.1,
            },
        }
        original_values = search_space["stdp"]["tau_pre"].copy()
        resolve_search_space(search_space)

        assert search_space["stdp"]["tau_pre"] == original_values


class TestFindHyperparameters:
    def test_returns_best_config_and_saves_json(self, tmp_path):
        """With deterministic run_experiment, best config is the one with higher accuracy."""

        def run_experiment(config):
            # Higher learning_rate → higher accuracy in our fake experiment
            lr = config["stdp"]["learning_rate"]
            acc = lr * 10  # lr=0.1 → acc=1.0, lr=0.01 → acc=0.1
            return {
                "train": {"accuracy": acc, "precision": acc, "recall": acc, "f1": acc},
                "validation": {
                    "accuracy": acc,
                    "precision": acc,
                    "recall": acc,
                    "f1": acc,
                },
            }

        search_space = {
            "stdp": {
                "tau_pre": 0.1,
                "learning_rate": [0.01, 0.1],
            },
        }

        exp_name = "test_find_hp"
        best = find_hyperparameters(
            run_experiment=run_experiment,
            search_space=search_space,
            exp_name=exp_name,
            seeds=[1, 2],
        )

        try:
            # Best config should be the one with learning_rate=0.1
            assert best["stdp"]["learning_rate"] == 0.1

            # JSON should have been saved
            results_path = f"logs/{exp_name}/search_results.json"
            assert os.path.exists(results_path)
            with open(results_path) as f:
                results = json.load(f)
            assert len(results) == 2
            # First result should be the best (highest val accuracy)
            assert results[0]["config"]["stdp"]["learning_rate"] == 0.1
        finally:
            import shutil

            shutil.rmtree(f"logs/{exp_name}", ignore_errors=True)
