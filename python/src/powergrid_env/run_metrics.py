"""TensorBoard scalar access shared by run_report.py and orchestrate.py."""

import glob
import os

from tensorboard.backend.event_processing import event_accumulator


def tb_dirs(run_dir: str) -> list[str]:
    """TensorBoard run subdirectories under <run_dir>/tb, oldest → newest."""
    return sorted(
        (d for d in glob.glob(os.path.join(run_dir, "tb", "*")) if os.path.isdir(d)),
        key=os.path.getmtime,
    )


def load_tb_series(run_dir: str) -> dict[str, list]:
    """All scalar series of the newest tb run: tag -> list of ScalarEvent
    (``.step``, ``.value``, ``.wall_time``). Empty dict if no event files."""
    dirs = tb_dirs(run_dir)
    if not dirs:
        return {}
    ea = event_accumulator.EventAccumulator(dirs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return {t: ea.Scalars(t) for t in ea.Tags()["scalars"]}
