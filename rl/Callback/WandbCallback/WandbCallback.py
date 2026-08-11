from __future__ import annotations

from typing import Any

import wandb as wandb_lib

from rl_tools.rl.Callback.SinkCallback import SinkCallback


class WandbCallback(SinkCallback):
    """Write blackboard scalars/histograms to a Weights & Biases run.

    Full per-metric history is stored natively by W&B (each scalar is logged
    with its step), so complete training curves are available in the dashboard
    without keeping a local chart buffer. If the blackboard carries an
    ``eval/latest`` dict, it is logged as a ``wandb.Table``.
    """

    def __init__(
        self, wandb_run: wandb_lib.Run, *, cursor_name: str | None = None
    ) -> None:
        super().__init__(cursor_name=cursor_name)
        if wandb_run is None:
            raise ValueError("wandb_run is required for WandbCallback")
        self.wandb_run = wandb_run

    def _write_scalar(self, step: int, key: str, value: float) -> None:
        self.wandb_run.log({key: value}, step=step)

    def _write_histogram(self, step: int, key: str, values: Any) -> None:
        self.wandb_run.log({key: wandb_lib.Histogram(values)}, step=step)

    def _flush(self) -> None:
        super()._flush()
        if self.agent is None:
            return
        latest = self.agent.blackboard.get("eval/latest")
        if latest:
            columns = list(latest.keys())
            row = [latest[c] for c in columns]
            table = wandb_lib.Table(columns=columns, data=[row])
            self.wandb_run.log({"eval/latest": table})
