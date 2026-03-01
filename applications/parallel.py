import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

from applications.common import merge_seed_results


@dataclass
class Task:
    """A single experiment run as a subprocess command."""

    command: list[str]
    group_key: str
    label: str


@dataclass
class Group:
    """A set of tasks whose results are merged after all succeed."""

    key: str
    merge_dir: str
    tasks: list[Task]


def _run_task(
    task: Task,
    on_epoch: Callable | None = None,
) -> subprocess.CompletedProcess:
    """Run a single task as a subprocess, streaming stdout for epoch markers.

    When on_epoch is provided, stdout is read line-by-line and the callback
    is invoked for each line starting with 'EPOCH '.
    """
    if on_epoch is None:
        return subprocess.run(task.command, capture_output=True, text=True)

    proc = subprocess.Popen(
        task.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Read stderr in a separate thread to avoid pipe deadlocks.
    stderr_chunks: list[str] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read())
    )
    stderr_thread.start()

    stdout_lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        stdout_lines.append(line)
        if line.startswith("EPOCH "):
            on_epoch()

    proc.wait()
    stderr_thread.join()
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        "".join(stdout_lines),
        "".join(stderr_chunks),
    )


def run_experiment_groups(
    groups: list[Group],
    *,
    max_workers: int = 10,
    description: str = "Experiments",
    epochs_per_task: int | None = None,
) -> None:
    """Execute all tasks across groups with bounded parallelism and tqdm progress.

    Merges results for each group when all its tasks complete successfully.
    Skips merge for groups with any failed task.
    Exits with code 1 if any task failed.

    When epochs_per_task is set, subprocesses are expected to print 'EPOCH k/n'
    lines to stdout.  Completed epochs are shown in the tqdm postfix.
    """
    if not groups:
        return

    # Build lookup structures
    all_tasks = []
    group_total: dict[str, int] = {}
    group_done: dict[str, int] = {}
    group_failed: dict[str, bool] = {}
    group_merge_dir: dict[str, str] = {}
    lock = threading.Lock()
    errors: list[tuple[str, str]] = []

    for group in groups:
        group_total[group.key] = len(group.tasks)
        group_done[group.key] = 0
        group_failed[group.key] = False
        group_merge_dir[group.key] = group.merge_dir
        all_tasks.extend(group.tasks)

    total_epochs = len(all_tasks) * epochs_per_task if epochs_per_task else None
    completed_epochs = [0]  # list for closure mutability

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(all_tasks), desc=description) as pbar:
            epoch_cb = None
            if epochs_per_task is not None:

                def epoch_cb():
                    with lock:
                        completed_epochs[0] += 1
                        pbar.set_postfix_str(
                            f"epochs: {completed_epochs[0]}/{total_epochs}"
                        )

            future_to_task = {
                pool.submit(_run_task, task, on_epoch=epoch_cb): task
                for task in all_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                pbar.update(1)

                if result.returncode != 0:
                    with lock:
                        group_failed[task.group_key] = True
                        errors.append((task.label, result.stderr))
                        group_done[task.group_key] += 1
                else:
                    with lock:
                        group_done[task.group_key] += 1
                        gk = task.group_key
                        if group_done[gk] == group_total[gk] and not group_failed[gk]:
                            merge_seed_results(group_merge_dir[gk])

    if errors:
        print("\nFailed tasks:", file=sys.stderr)
        for label, stderr in errors:
            print(f"  {label}:", file=sys.stderr)
            if stderr.strip():
                for line in stderr.strip().splitlines():
                    print(f"    {line}", file=sys.stderr)
        sys.exit(1)
