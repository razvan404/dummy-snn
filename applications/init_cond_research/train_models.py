import argparse
import math
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from applications.common import merge_seed_results
from applications.datasets import DATASETS, create_dataset
from applications.deep_linear.progress_callbacks import make_progress_callbacks
from applications.threshold_research.train_models import train_with_metrics

PROGRESS_LOG_INTERVAL = 200

SEED_START = 1
DEFAULT_NUM_SEEDS = 5
T_OBJECTIVES = [round(0.4 + v * 0.05, 2) for v in range(12)] + [0.875]  # 0.4 to 0.95
THRESHOLD_OFFSETS = [-75, -60, -40, -20, -10, 0, 10, 20, 40, 60, 75, 100]

_worker_state = {}


def _init_worker(dataset_name, worker_id_counter, progress_queue):
    """Called once per worker process to load the dataset."""
    with worker_id_counter.get_lock():
        wid = worker_id_counter.value
        worker_id_counter.value += 1
    train_loader, val_loader = create_dataset(dataset_name)
    _worker_state["loaders"] = (train_loader, val_loader)
    _worker_state["spike_shape"] = (2, *train_loader.dataset.image_shape)
    _worker_state["steps"] = {"train": len(train_loader), "val": len(val_loader)}
    _worker_state["worker_id"] = wid
    _worker_state["progress_queue"] = progress_queue


def _train_single(thresh, t_obj, seed, output_dir, num_epochs):
    """Run one training config in a worker process."""
    wid = _worker_state["worker_id"]
    queue = _worker_state["progress_queue"]
    steps = _worker_state["steps"]
    label = f"thresh={thresh} tobj={t_obj} seed={seed}"
    epoch = [1]

    def on_batch_end(idx, _dw, split):
        if idx % PROGRESS_LOG_INTERVAL == 0:
            steps_str = f"{idx + 1}/{steps[split]}" if split in steps else f"{idx + 1}"
            queue.put(
                (
                    wid,
                    f"{label} epoch={epoch[0]}/{num_epochs} {split} {steps_str}",
                )
            )

    def on_epoch_end(e, _total):
        epoch[0] = e + 1

    train_with_metrics(
        dataset_loaders=_worker_state["loaders"],
        spike_shape=_worker_state["spike_shape"],
        seed=seed,
        avg_threshold=thresh,
        output_dir=output_dir,
        num_epochs=num_epochs,
        t_objective=t_obj,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
    )
    queue.put((wid, ""))


def _collect_tasks(thresholds, t_objectives, seeds, dataset, force):
    """Collect all (thresh, t_obj, seed, output_dir) tuples that need to run."""
    tasks = []
    skipped = 0
    for thresh in thresholds:
        for t_obj in t_objectives:
            base_dir = (
                f"logs/{dataset}/init_cond_research" f"/thresh_{thresh}/tobj_{t_obj}"
            )
            for seed in seeds:
                output_dir = f"{base_dir}/seed_{seed}"
                if not force and os.path.exists(f"{output_dir}/metrics.json"):
                    skipped += 1
                    continue
                tasks.append((thresh, t_obj, seed, output_dir))
    return tasks, skipped


def _run_sequential(tasks, dataset_loaders, spike_shape, num_epochs, skipped):
    """Run all tasks sequentially in the current process."""
    train_loader, val_loader = dataset_loaders
    steps = {"train": len(train_loader), "val": len(val_loader)}
    total = len(tasks) + skipped
    with tqdm(
        total=total, desc="Init cond research: training", initial=skipped
    ) as pbar:
        for thresh, t_obj, seed, output_dir in tasks:
            label = f"thresh={thresh} t_obj={t_obj} seed={seed}"
            pbar.set_postfix_str(label)
            on_batch_end, on_epoch_end = make_progress_callbacks(
                pbar,
                label,
                num_epochs,
                steps,
            )
            train_with_metrics(
                dataset_loaders=dataset_loaders,
                spike_shape=spike_shape,
                seed=seed,
                avg_threshold=thresh,
                output_dir=output_dir,
                num_epochs=num_epochs,
                t_objective=t_obj,
                on_batch_end=on_batch_end,
                on_epoch_end=on_epoch_end,
            )
            pbar.update(1)


def _progress_reader(queue, worker_bars, stop_event):
    """Read progress updates from worker processes and render in main process."""
    while not stop_event.is_set() or not queue.empty():
        try:
            wid, msg = queue.get(timeout=0.1)
            if wid in worker_bars:
                worker_bars[wid].set_postfix_str(msg)
        except Exception:
            continue


def _run_parallel(tasks, dataset, num_epochs, num_workers, skipped):
    """Run all tasks across multiple worker processes."""
    total = len(tasks) + skipped
    worker_id_counter = multiprocessing.Value("i", 0)
    progress_queue = multiprocessing.Queue()

    overall_bar = tqdm(
        total=total,
        desc="Init cond research: training",
        initial=skipped,
        position=0,
    )
    worker_bars = {
        i: tqdm(total=0, position=i + 1, bar_format="{postfix}", leave=False)
        for i in range(num_workers)
    }

    stop_event = threading.Event()
    reader = threading.Thread(
        target=_progress_reader,
        args=(progress_queue, worker_bars, stop_event),
        daemon=True,
    )
    reader.start()

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(dataset, worker_id_counter, progress_queue),
        ) as pool:
            futures = {
                pool.submit(
                    _train_single,
                    thresh,
                    t_obj,
                    seed,
                    output_dir,
                    num_epochs,
                ): (thresh, t_obj, seed)
                for thresh, t_obj, seed, output_dir in tasks
            }
            for future in as_completed(futures):
                future.result()
                overall_bar.update(1)
    finally:
        stop_event.set()
        reader.join()
        overall_bar.close()
        for bar in worker_bars.values():
            bar.close()


def run(
    dataset: str,
    *,
    num_epochs: int = 30,
    force: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    num_workers: int = 1,
):
    seeds = list(range(SEED_START, SEED_START + num_seeds))
    train_loader, val_loader = create_dataset(dataset)
    spike_shape = (2, *train_loader.dataset.image_shape)
    base_threshold = math.prod(spike_shape) / 20
    thresholds = [base_threshold + offset for offset in THRESHOLD_OFFSETS]

    tasks, skipped = _collect_tasks(thresholds, T_OBJECTIVES, seeds, dataset, force)
    if skipped:
        tqdm.write(f"Skipping {skipped} already-complete runs")

    if num_workers <= 1:
        _run_sequential(
            tasks,
            (train_loader, val_loader),
            spike_shape,
            num_epochs,
            skipped,
        )
    else:
        # Workers load their own datasets; free main-process copy
        del train_loader, val_loader
        _run_parallel(tasks, dataset, num_epochs, num_workers, skipped)

    # Merge seed results for all (thresh, t_obj) groups
    for thresh in thresholds:
        for t_obj in T_OBJECTIVES:
            base_dir = (
                f"logs/{dataset}/init_cond_research" f"/thresh_{thresh}/tobj_{t_obj}"
            )
            merge_seed_results(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models across initial thresholds and t-objectives"
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run(
        args.dataset,
        num_epochs=args.epochs,
        force=args.force,
        num_seeds=args.seeds,
        num_workers=args.workers,
    )
