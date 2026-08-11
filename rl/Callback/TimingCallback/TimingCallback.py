from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from tensordict import TensorDict

from rl_tools.rl.Callback.Callback import Callback


class TimingCallback(Callback):
    """Time PPO train-loop segments and log ms per env-step."""

    def __init__(self, *, log_prefix: str = "timing") -> None:
        super().__init__()
        self.log_prefix = log_prefix
        self._t_rollout_start: float | None = None
        self._t_rollout_end: float | None = None
        self._t_update_start: float | None = None
        self._rollout_s: float | None = None
        self._gae_s: float | None = None
        self._env_steps: int = 0

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._t_rollout_start = time.perf_counter()
        self._t_rollout_end = None
        self._t_update_start = None
        self._rollout_s = None
        self._gae_s = None
        self._env_steps = 0

    def on_rollout_end(self, rollout: TensorDict) -> None:
        now = time.perf_counter()
        self._t_rollout_end = now
        self._env_steps = int(rollout["rewards"].numel())
        if self._t_rollout_start is None:
            return
        self._rollout_s = now - self._t_rollout_start
        self._log_segment("rollout", self._rollout_s)

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
        now = time.perf_counter()
        self._t_update_start = now
        if self._t_rollout_end is None:
            return
        self._gae_s = now - self._t_rollout_end
        self._log_segment("gae", self._gae_s)

    def on_update_end(self, update_info: dict) -> None:
        if self._t_update_start is None:
            return
        update_s = time.perf_counter() - self._t_update_start
        self._log_segment("update", update_s)
        parts = [self._rollout_s, self._gae_s, update_s]
        if any(p is None for p in parts):
            return
        iteration_s = float(self._rollout_s) + float(self._gae_s) + update_s
        self._log_segment("iteration", iteration_s)

    def _log_segment(self, name: str, seconds: float) -> None:
        if self.agent is None:
            return
        prefix = self.log_prefix
        self.agent.blackboard.record(
            f"{prefix}/{name}_s", seconds, self.agent.global_step
        )
        if self._env_steps > 0:
            self.agent.blackboard.record(
                f"{prefix}/{name}_ms_per_env_step",
                1000.0 * seconds / self._env_steps,
                self.agent.global_step,
            )
