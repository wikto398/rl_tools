from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback


class NetworkSaveCallback(Callback):
    """Save agent checkpoints on train end and optionally every N updates."""

    def __init__(
        self,
        save_path: str,
        *,
        save_every_updates: int | None = None,
    ) -> None:
        super().__init__()
        if not save_path:
            raise ValueError("save_path must be a non-empty string")
        if save_every_updates is not None and save_every_updates <= 0:
            raise ValueError(
                f"save_every_updates must be positive when set, got {save_every_updates}"
            )
        self.save_path = save_path
        self.save_every_updates = save_every_updates
        self._updates_since_save = 0

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        self._save(self.save_path)

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
        if self.save_every_updates is None:
            return
        self._updates_since_save += 1
        if self._updates_since_save < self.save_every_updates:
            return
        self._updates_since_save = 0
        latest_path = self._latest_path()
        self._save(latest_path)

    def _latest_path(self) -> str:
        path = Path(self.save_path)
        return str(path.with_name(f"{path.stem}_latest{path.suffix}"))

    def _save(self, path: str) -> None:
        if self.agent is None:
            raise ValueError(
                "Agent is not set. Ensure the callback is attached to an agent."
            )
        self.agent.info(f"NetworkSaveCallback: saving checkpoint to {path}")
        self.agent.save(path)
