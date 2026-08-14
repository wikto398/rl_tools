from __future__ import annotations

import csv
from typing import Any

from rl_tools.rl.Callback.SinkCallback import SinkCallback


class CsvCallback(SinkCallback):
    """Dump every blackboard scalar to a long-format CSV (one row per
    ``step, key, value``).

    Enable with ``--csv_metrics`` (off by default); writes
    ``<log_path>/metrics.csv``. Histograms are skipped (scalars only).
    Load with ``pandas.read_csv`` and filter, e.g.
    ``df[df.key == "eval/win_rate"]`` or
    ``df.pivot(index="step", columns="key", values="value")``.
    """

    def __init__(self, *, path: str) -> None:
        super().__init__()
        # Handle must live across the run (opened here, closed in _on_train_end).
        self._f = open(path, "w", buffering=1)  # noqa: SIM115
        self._writer = csv.writer(self._f)
        self._writer.writerow(["step", "key", "value"])

    def _write_scalar(self, step: int, key: str, value: float) -> None:
        self._writer.writerow([step, key, value])

    def _write_histogram(self, step: int, key: str, values: Any) -> None:
        pass

    def _on_train_end(self) -> None:
        self._f.close()
