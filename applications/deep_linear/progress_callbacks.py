from collections.abc import Callable

PROGRESS_LOG_INTERVAL = 200


def make_progress_callbacks(
    pbar,
    label: str,
    num_epochs: int,
    step_counts: dict[str, int] | None = None,
) -> tuple[Callable, Callable]:
    """Create on_batch_end and on_epoch_end callbacks for a tqdm progress bar.

    pbar: tqdm progress bar instance.
    label: string identifying the current run (e.g. "thresh=12.8 seed=1").
    num_epochs: total number of epochs.
    step_counts: optional dict mapping split names to step counts (e.g. {"train": 600}).
    """
    epoch = [1]

    def on_batch_end(idx, dw, split):
        if idx % PROGRESS_LOG_INTERVAL == 0:
            steps_str = (
                f"{idx + 1}/{step_counts[split]}"
                if step_counts and split in step_counts
                else f"{idx + 1}"
            )
            pbar.set_postfix_str(
                f"{label} epoch={epoch[0]}/{num_epochs} {split} {steps_str}"
            )

    def on_epoch_end(e, _total):
        epoch[0] = e + 1

    return on_batch_end, on_epoch_end
