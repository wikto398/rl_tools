from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback

if TYPE_CHECKING:
    from rl_tools.rl.RLAgent import RLAgent


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        super().__init__()
        self.callbacks = list(callbacks or [])

    def setup(self, agent: RLAgent) -> None:
        super().setup(agent)
        for callback in self.callbacks:
            callback.setup(agent)

    def on_train_start(self) -> None:
        for callback in self.callbacks:
            callback.on_train_start()

    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end()

    def on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def on_rollout_end(self, rollout: TensorDict) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end(rollout)

    def on_step(
        self,
        *,
        actions: Any,
        rewards: Sequence[float],
        dones: Sequence[bool],
        infos: Sequence[dict],
    ) -> bool:
        continue_training = True
        for callback in self.callbacks:
            continue_training = (
                callback.on_step(
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    infos=infos,
                )
                and continue_training
            )
        return continue_training

    def on_update_start(self, rollout: TensorDict) -> None:
        for callback in self.callbacks:
            callback.on_update_start(rollout)

    def on_update_end(self, update_info: dict) -> None:
        for callback in self.callbacks:
            callback.on_update_end(update_info)
