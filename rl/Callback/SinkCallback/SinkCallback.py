from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback

if TYPE_CHECKING:
    from rl_tools.rl.RLAgent import RLAgent


class SinkCallback(Callback):
    """Base class for callbacks that consume the agent blackboard.

    Subclasses register a cursor, drain new events on each update (and at
    train end), and implement ``_write_scalar`` / ``_write_histogram`` to push
    values to a specific sink (console, TensorBoard, W&B, ...). All the shared
    plumbing — cursor lifecycle, draining, and the six abstract ``Callback``
    hooks — lives here so adding a new sink only requires two write methods.
    """

    def __init__(self, *, cursor_name: str | None = None) -> None:
        super().__init__()
        self.cursor_name = cursor_name or type(self).__name__.lower()

    def setup(self, agent: RLAgent) -> None:
        super().setup(agent)
        agent.blackboard.register_cursor(self.cursor_name)

    @abstractmethod
    def _write_scalar(self, step: int, key: str, value: float) -> None: ...

    @abstractmethod
    def _write_histogram(self, step: int, key: str, values: Any) -> None: ...

    def _flush(self) -> None:
        if self.agent is None:
            return
        for event in self.agent.blackboard.drain(self.cursor_name):
            if event.kind == "histogram":
                values = event.value
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                values = np.asarray(values).ravel()
                if values.size == 0:
                    continue
                self._write_histogram(event.step, event.key, values)
            else:
                self._write_scalar(event.step, event.key, float(event.value))

    def _on_train_end(self) -> None:
        """Optional per-sink cleanup after the final flush."""

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        self._flush()
        self._on_train_end()

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self, rollout: TensorDict) -> None:
        pass

    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        return True

    def on_update_start(self, rollout: TensorDict) -> None:
        pass

    def on_update_end(self, update_info: dict) -> None:
        self._flush()
