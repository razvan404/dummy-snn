import subprocess
import sys
from unittest.mock import patch, call

import pytest

from applications.parallel import Group, Task, run_experiment_groups


class TestTaskDataclass:
    def test_stores_command_and_label(self):
        task = Task(
            command=[sys.executable, "-c", "print('hello')"],
            group_key="g1",
            label="test task",
        )
        assert task.command == [sys.executable, "-c", "print('hello')"]
        assert task.group_key == "g1"
        assert task.label == "test task"


class TestGroupDataclass:
    def test_stores_key_and_merge_dir_and_tasks(self):
        task = Task(command=["echo"], group_key="g1", label="t1")
        group = Group(key="g1", merge_dir="/tmp/merge", tasks=[task])
        assert group.key == "g1"
        assert group.merge_dir == "/tmp/merge"
        assert group.tasks == [task]


class TestRunExperimentGroups:
    def test_runs_all_tasks_successfully(self):
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g1",
                        label="task 1",
                    ),
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g1",
                        label="task 2",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results") as mock_merge:
            run_experiment_groups(groups, max_workers=2)
            mock_merge.assert_called_once_with("/tmp/fake")

    def test_skips_merge_when_any_task_in_group_fails(self):
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g1",
                        label="ok task",
                    ),
                    Task(
                        command=[sys.executable, "-c", "raise SystemExit(1)"],
                        group_key="g1",
                        label="fail task",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results") as mock_merge:
            with pytest.raises(SystemExit):
                run_experiment_groups(groups, max_workers=2)
            mock_merge.assert_not_called()

    def test_merges_each_group_independently(self):
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/g1",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g1",
                        label="g1 task",
                    ),
                ],
            ),
            Group(
                key="g2",
                merge_dir="/tmp/g2",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g2",
                        label="g2 task",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results") as mock_merge:
            run_experiment_groups(groups, max_workers=2)
            assert mock_merge.call_count == 2
            mock_merge.assert_any_call("/tmp/g1")
            mock_merge.assert_any_call("/tmp/g2")

    def test_exits_with_code_1_on_failure(self):
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "raise SystemExit(1)"],
                        group_key="g1",
                        label="fail",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results"):
            with pytest.raises(SystemExit) as exc_info:
                run_experiment_groups(groups, max_workers=2)
            assert exc_info.value.code == 1

    def test_healthy_group_merges_even_if_other_group_fails(self):
        groups = [
            Group(
                key="good",
                merge_dir="/tmp/good",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="good",
                        label="ok",
                    ),
                ],
            ),
            Group(
                key="bad",
                merge_dir="/tmp/bad",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "raise SystemExit(1)"],
                        group_key="bad",
                        label="fail",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results") as mock_merge:
            with pytest.raises(SystemExit):
                run_experiment_groups(groups, max_workers=2)
            mock_merge.assert_called_once_with("/tmp/good")

    def test_respects_max_workers(self):
        """Verify ThreadPoolExecutor is created with the given max_workers."""
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[sys.executable, "-c", "pass"],
                        group_key="g1",
                        label="t1",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results"):
            with patch(
                "applications.parallel.ThreadPoolExecutor",
                wraps=__import__("concurrent.futures").futures.ThreadPoolExecutor,
            ) as mock_pool:
                run_experiment_groups(groups, max_workers=7)
                mock_pool.assert_called_once_with(max_workers=7)

    def test_no_tasks_is_a_noop(self):
        """Empty group list should complete without errors."""
        with patch("applications.parallel.merge_seed_results") as mock_merge:
            run_experiment_groups([], max_workers=2)
            mock_merge.assert_not_called()

    def test_captures_stderr_on_failure(self, capsys):
        """Failed tasks should show stderr in the error report."""
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[
                            sys.executable,
                            "-c",
                            "import sys; sys.stderr.write('boom\\n'); sys.exit(1)",
                        ],
                        group_key="g1",
                        label="noisy fail",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results"):
            with pytest.raises(SystemExit):
                run_experiment_groups(groups, max_workers=2)
        captured = capsys.readouterr()
        assert "boom" in captured.err or "boom" in captured.out


class TestRunTaskEpochReporting:
    def test_calls_on_epoch_for_epoch_lines(self):
        """_run_task should invoke the on_epoch callback for each EPOCH line."""
        from applications.parallel import _run_task

        task = Task(
            command=[
                sys.executable,
                "-c",
                "print('EPOCH 1/3', flush=True); print('EPOCH 2/3', flush=True); print('EPOCH 3/3', flush=True)",
            ],
            group_key="g1",
            label="t1",
        )
        epoch_count = []
        result = _run_task(task, on_epoch=lambda: epoch_count.append(1))
        assert len(epoch_count) == 3
        assert result.returncode == 0

    def test_no_callback_still_works(self):
        """_run_task without on_epoch should work normally."""
        from applications.parallel import _run_task

        task = Task(
            command=[sys.executable, "-c", "print('EPOCH 1/2')"],
            group_key="g1",
            label="t1",
        )
        result = _run_task(task)
        assert result.returncode == 0

    def test_captures_all_stdout(self):
        """_run_task should still capture full stdout when streaming."""
        from applications.parallel import _run_task

        task = Task(
            command=[
                sys.executable,
                "-c",
                "print('hello'); print('EPOCH 1/1'); print('world')",
            ],
            group_key="g1",
            label="t1",
        )
        result = _run_task(task, on_epoch=lambda: None)
        assert "hello" in result.stdout
        assert "EPOCH 1/1" in result.stdout
        assert "world" in result.stdout

    def test_captures_stderr_when_streaming(self):
        """_run_task should still capture stderr properly."""
        from applications.parallel import _run_task

        task = Task(
            command=[
                sys.executable,
                "-c",
                "import sys; print('EPOCH 1/1'); sys.stderr.write('err\\n'); sys.exit(1)",
            ],
            group_key="g1",
            label="t1",
        )
        result = _run_task(task, on_epoch=lambda: None)
        assert result.returncode == 1
        assert "err" in result.stderr


class TestEpochProgressInGroups:
    def test_epochs_per_task_does_not_crash(self):
        """run_experiment_groups with epochs_per_task should complete normally."""
        groups = [
            Group(
                key="g1",
                merge_dir="/tmp/fake",
                tasks=[
                    Task(
                        command=[
                            sys.executable,
                            "-c",
                            "print('EPOCH 1/2', flush=True); print('EPOCH 2/2', flush=True)",
                        ],
                        group_key="g1",
                        label="t1",
                    ),
                ],
            ),
        ]
        with patch("applications.parallel.merge_seed_results"):
            run_experiment_groups(groups, max_workers=2, epochs_per_task=2)
