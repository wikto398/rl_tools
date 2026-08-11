from __future__ import annotations

from typing import Any

from torch.utils.tensorboard.writer import SummaryWriter

from rl_tools.rl.Callback.SinkCallback import SinkCallback


class TensorboardCallback(SinkCallback):
    """Write blackboard scalars/histograms to a TensorBoard SummaryWriter.

    Optionally registers an ``add_custom_scalars`` layout (multiline groups)
    on train start and closes the writer at train end.
    """

    def __init__(
        self,
        summary_writer: SummaryWriter,
        *,
        cursor_name: str | None = None,
        custom_scalars: dict | None = None,
    ) -> None:
        super().__init__(cursor_name=cursor_name)
        if summary_writer is None:
            raise ValueError("summary_writer is required for TensorboardCallback")
        self.summary_writer = summary_writer
        self.custom_scalars = custom_scalars or {}

    def on_train_start(self) -> None:
        if self.custom_scalars:
            self.summary_writer.add_custom_scalars(self.custom_scalars)

    def _write_scalar(self, step: int, key: str, value: float) -> None:
        self.summary_writer.add_scalar(key, value, global_step=step)

    def _write_histogram(self, step: int, key: str, values: Any) -> None:
        self.summary_writer.add_histogram(key, values, global_step=step)

    def _on_train_end(self) -> None:
        self.summary_writer.close()
