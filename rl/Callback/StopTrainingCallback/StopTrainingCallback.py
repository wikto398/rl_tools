from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback


class StopTrainingCallback(Callback):
    """Base callback that can request a graceful training stop."""

    def __init__(self) -> None:
        super().__init__()
        self._stopped = False

    def request_stop(self, reason: str | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.agent is not None:
            self.agent._stop_requested = True
            message = "StopTrainingCallback: stopping training"
            if reason:
                message = f"{message} ({reason})"
            self.agent.info(message)

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

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
        pass
